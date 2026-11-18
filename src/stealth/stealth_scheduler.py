# src/stealth/stealth_scheduler.py
"""
Stealth Scheduler — Integration Bridge.

Connects trained GAN Generator / RL Agent outputs to the distribution
pipeline (Scheduler + Dispatcher).  If no trained model is available
the system falls back to static NoiseController scheduling.
"""

from __future__ import annotations

import threading
import torch
from src.utils.checkpoint_io import load_trained_checkpoint
import numpy as np
import hashlib
from pathlib import Path
from typing import Literal, Optional

from src.distribution.noise import NoiseController
from src.distribution.profiles import ACTIVITY_PROFILES
from src.stealth.delay_contract import (  # noqa: F401
    GAN_DELAY_MAX,
    GAN_DELAY_MIN,
    GAN_DELAY_OFFSET,
    GAN_DELAY_SCALE,
    GAN_RAW_BAND,
    GAN_RAW_MAX,
    GAN_RAW_MIN,
    canonical_delay_mapping,
)
from src.stealth.rl.pacing import RL_DELAY_MIN

# Per-channel cooldowns, in seconds. The enterprise rate limits are
# ch0 = 10/min (6 s), ch1 = 5/min (12 s), ch2 = 15/min (4 s). This is the
# same table the Component 3 benchmark measures against and the RL
# environment's channel clocks encode, so every executor and the
# measurement share one constant.
CHANNEL_COOLDOWNS = {0: 6.0, 1: 12.0, 2: 4.0}

# Anchored to the project root so schedules work regardless of CWD.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_GAN_CHECKPOINT = _PROJECT_ROOT / "storage" / "models" / "gan_generator.pt"
DEFAULT_RL_CHECKPOINT = _PROJECT_ROOT / "storage" / "models" / "rl_agent.pt"
DEFAULT_ACTIVITY_CHECKPOINT = _PROJECT_ROOT / "storage" / "models" / "lsapp_activity_timing_train.json"
DEFAULT_TPP_CHECKPOINT = _PROJECT_ROOT / "storage" / "models" / "tpp_lsapp.pt"
DEFAULT_TPP_PPO_CHECKPOINT = _PROJECT_ROOT / "storage" / "models" / "tpp_ppo_lsapp.pt"

# Canonical WGAN delay mapping.
#
# The generator's delay head emits a softplus value, folded into the band
# of raw values the mapping can turn into the deployment window, so the
# mapping is never clipped and the gradient always reaches the head. The
# mapping below expands that band onto a human-plausible 0.5 s .. 30 s
# range and is the single definition used by the deployed GAN path, the
# deployed hybrid path, and the Component 3 benchmark, so the measured row
# and the shipped row cannot drift apart.
#
# The definition lives in `src.stealth.delay_contract`, which the
# generator also reads, and is re-exported here because the scheduler is
# where the deployment contract has always been pinned.


class StealthScheduler:
    """
    Unified scheduler that can operate in three modes:

    * **static**  – handcrafted NoiseController (always available)
    * **gan**     – TemporalPatternGenerator (needs trained checkpoint)
    * **rl**      – PPOAgent policy (needs trained checkpoint)

    If a checkpoint is requested but missing the scheduler automatically
    falls back to *static* mode and logs a warning.

    Delay convention: every mode returns `delays[i]` as the pause AFTER
    item i is sent — matching NoiseController (src/distribution/noise.py)
    and the API transmitter (src/api/server.py:_transmit_packets_sync),
    which both dispatch an item and THEN sleep for `delays[i]`. The RL
    environment (src/stealth/rl/environment.py) follows the same
    convention internally.

    Not re-entrant: `_rl_agent` and its `env`/`buffer` are lazily created
    once and reused across `schedule()` calls. `_schedule_rl` resets the
    env's episode state (queue, channels, transmission_history) on every
    call, but concurrent calls on the same StealthScheduler instance would
    race on that shared state. Serialize calls (e.g. one scheduler per
    request, or a lock around `schedule()`) if the API ever needs to serve
    concurrent /transmit requests.
    """

    def __init__(
        self,
        num_channels: int = 3,
        device: str = "cpu",
        profile: str = "casual",
        seed: int = 42,
    ):
        self.num_channels = num_channels
        self.device = device
        self.profile = profile
        self.seed = seed

        # Thread-safety for concurrent schedule() calls
        self._schedule_lock = threading.Lock()

        # Lazy-loaded models
        self._generator = None
        self._rl_agent = None
        self._warden = None
        self._activity_model = None
        self._activity_checkpoint_path: Optional[Path] = None
        self._activity_rng = np.random.default_rng(seed)
        self._tpp_model = None
        self._tpp_checkpoint_path: Optional[Path] = None
        self._tpp_ppo_agent = None
        self._tpp_ppo_checkpoint_path: Optional[Path] = None
        self._tpp_rng = np.random.default_rng(seed + 17)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        media_ids: list[str],
        mode: Literal["static", "gan", "rl", "hybrid", "activity", "tpp", "tpp_ppo", "auto"] = "static",
        base_delay: float = 3.0,
        gan_checkpoint: Optional[Path] = None,
        rl_checkpoint: Optional[Path] = None,
        time_of_day: Optional[float] = None,
        activity_checkpoint: Optional[Path] = None,
        activity_pause_cap_s: float = 45.0,
        tpp_checkpoint: Optional[Path] = None,
        tpp_ppo_checkpoint: Optional[Path] = None,
        tpp_pause_cap_s: Optional[float] = None,
    ) -> dict:
        """
        Produce a timed schedule for *media_ids*.

        mode="auto" runs the rl -> gan -> static cascade, matching the
        documented behaviour in scripts/runtime/run_sender.py.
        mode="hybrid" combines WGAN timing pauses with closed-loop channel
        cooldown tracking so every exported landing respects the per-channel
        cooldowns the Component 3 benchmark measures.

        time_of_day pins the hour-of-day conditioning handed to the WGAN
        generator (0-23). When None, the GAN and hybrid paths draw a random
        hour. Pinning it is what lets the Component 3 benchmark drive the
        deployed path under a controlled hour instead of re-implementing it.

        Returns
        -------
        dict with keys:
            items    – list[str]  media IDs after noise filtering
            delays   – list[float]  inter-item delays in seconds
            channels – list[int]  channel index per item
            mode_used – str  actual mode that ran (may differ from *mode* on fallback)
        """
        # Thread-safe scheduling: serialize concurrent calls
        with self._schedule_lock:
            return self._schedule_unlocked(
                media_ids,
                mode,
                base_delay,
                gan_checkpoint,
                rl_checkpoint,
                time_of_day,
                activity_checkpoint,
                activity_pause_cap_s,
                tpp_checkpoint,
                tpp_ppo_checkpoint,
                tpp_pause_cap_s,
            )

    @staticmethod
    def _validate_schedule(schedule: dict, media_ids: list[str]) -> dict:
        """
        Enforce the payload contract: the schedule must carry exactly one
        (delay, channel) per media ID the caller asked to send (media IDs
        already encode the RS-ECC'd payload via SemanticEncoder — dropping
        or duplicating one here would corrupt the message), and no delay
        may be negative.
        """
        n = len(media_ids)
        if len(schedule["items"]) != n or len(schedule["delays"]) != n or len(
            schedule["channels"]
        ) != n:
            raise ValueError(
                f"StealthScheduler produced a schedule of length "
                f"{len(schedule['delays'])} for {n} media_ids "
                f"(mode_used={schedule['mode_used']}) — refusing to transmit "
                f"a schedule that doesn't match the encoded payload length."
            )
        if any(d < 0 for d in schedule["delays"]):
            raise ValueError(
                f"StealthScheduler produced a negative delay "
                f"(mode_used={schedule['mode_used']})."
            )
        return schedule

    def _schedule_unlocked(
        self,
        media_ids: list[str],
        mode: Literal["static", "gan", "rl", "hybrid", "activity", "tpp", "tpp_ppo", "auto"],
        base_delay: float,
        gan_checkpoint: Optional[Path],
        rl_checkpoint: Optional[Path],
        time_of_day: Optional[float] = None,
        activity_checkpoint: Optional[Path] = None,
        activity_pause_cap_s: float = 45.0,
        tpp_checkpoint: Optional[Path] = None,
        tpp_ppo_checkpoint: Optional[Path] = None,
        tpp_pause_cap_s: Optional[float] = None,
    ) -> dict:
        """Internal scheduling logic (called under lock)."""
        if mode == "auto":
            schedule = self._schedule_rl(
                media_ids,
                base_delay,
                self._resolve_checkpoint(rl_checkpoint, DEFAULT_RL_CHECKPOINT),
            )
            if schedule["mode_used"] != "rl":
                schedule = self._schedule_gan(
                    media_ids,
                    base_delay,
                    self._resolve_checkpoint(gan_checkpoint, DEFAULT_GAN_CHECKPOINT),
                    time_of_day,
                )
            if schedule["mode_used"] not in ("rl", "gan"):
                schedule = self._schedule_static(media_ids, base_delay)
            return self._validate_schedule(schedule, media_ids)

        if mode == "hybrid":
            checkpoint = self._resolve_checkpoint(
                gan_checkpoint, DEFAULT_GAN_CHECKPOINT
            )
            schedule = self._schedule_hybrid(
                media_ids, base_delay, checkpoint, time_of_day
            )
        elif mode == "gan":
            checkpoint = self._resolve_checkpoint(
                gan_checkpoint, DEFAULT_GAN_CHECKPOINT
            )
            schedule = self._schedule_gan(
                media_ids, base_delay, checkpoint, time_of_day
            )
        elif mode == "rl":
            checkpoint = self._resolve_checkpoint(rl_checkpoint, DEFAULT_RL_CHECKPOINT)
            schedule = self._schedule_rl(media_ids, base_delay, checkpoint)
        elif mode == "activity":
            checkpoint = self._resolve_checkpoint(
                activity_checkpoint, DEFAULT_ACTIVITY_CHECKPOINT
            )
            schedule = self._schedule_activity(
                media_ids, checkpoint, pause_cap_s=activity_pause_cap_s
            )
        elif mode == "tpp":
            checkpoint = self._resolve_checkpoint(tpp_checkpoint, DEFAULT_TPP_CHECKPOINT)
            schedule = self._schedule_tpp(
                media_ids,
                base_delay,
                checkpoint,
                time_of_day=time_of_day,
                pause_cap_s=tpp_pause_cap_s,
            )
        elif mode == "tpp_ppo":
            checkpoint = self._resolve_checkpoint(
                tpp_ppo_checkpoint, DEFAULT_TPP_PPO_CHECKPOINT
            )
            schedule = self._schedule_tpp_ppo(
                media_ids,
                base_delay,
                checkpoint,
                time_of_day=time_of_day,
                pause_cap_s=tpp_pause_cap_s,
            )
        else:
            schedule = self._schedule_static(media_ids, base_delay)

        return self._validate_schedule(schedule, media_ids)

    @staticmethod
    def _resolve_checkpoint(explicit: Optional[Path], default: Path) -> Optional[Path]:
        """Prefer an explicit path; otherwise use the project-root default if present."""
        if explicit is not None:
            return Path(explicit)
        return default if default.exists() else None

    def _load_activity_model(self, checkpoint: Optional[Path]):
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "activity timing mode requires a fitted JSON checkpoint; "
                "pass activity_checkpoint or create the default checkpoint"
            )
        checkpoint = Path(checkpoint).resolve()
        if self._activity_model is not None and self._activity_checkpoint_path == checkpoint:
            return self._activity_model
        from src.stealth.activity_timing import ContiguousGapBlockModel

        self._activity_model = ContiguousGapBlockModel.load(checkpoint)
        self._activity_checkpoint_path = checkpoint
        return self._activity_model

    def _schedule_activity(
        self,
        media_ids: list[str],
        checkpoint: Optional[Path],
        *,
        pause_cap_s: float,
    ) -> dict:
        """Generate a contributor-balanced full-gap timing schedule from a fitted model."""
        if not np.isfinite(pause_cap_s) or pause_cap_s <= 0:
            raise ValueError("activity_pause_cap_s must be finite and positive")
        model = self._load_activity_model(checkpoint)
        sampled = model.sample_delays(len(media_ids), self._activity_rng)
        capped = np.minimum(
            np.asarray(sampled["delays"], dtype=np.float64), float(pause_cap_s)
        ).tolist()
        from src.stealth.activity_timing import route_gaps_with_cooldowns

        cooldowns = {
            channel: CHANNEL_COOLDOWNS.get(channel, 6.0)
            for channel in range(self.num_channels)
        }
        delays, channels = route_gaps_with_cooldowns(
            capped, self._activity_rng, cooldowns
        )
        contributor_hash = hashlib.sha256(
            str(sampled["contributor_id"]).encode("utf-8")
        ).hexdigest()[:16]
        return {
            "items": media_ids,
            "delays": delays,
            "channels": channels,
            "mode_used": "activity",
            "activity_source_contributor_sha256": contributor_hash,
            "activity_pause_cap_s": float(pause_cap_s),
        }

    # ------------------------------------------------------------------
    # Autoregressive Log-Normal Mixture TPP & Prior-Guided PPO modes
    # ------------------------------------------------------------------

    def _load_tpp_model(self, checkpoint: Optional[Path]):
        if checkpoint is None or not Path(checkpoint).is_file():
            return None
        resolved = Path(checkpoint).resolve()
        if self._tpp_model is not None and self._tpp_checkpoint_path == resolved:
            return self._tpp_model
        from src.stealth.tpp.model import TemporalPointProcessGenerator

        self._tpp_model = TemporalPointProcessGenerator.load_checkpoint(
            resolved, device=self.device
        )
        self._tpp_checkpoint_path = resolved
        return self._tpp_model

    def _schedule_tpp(
        self,
        media_ids: list[str],
        base_delay: float,
        checkpoint: Optional[Path],
        *,
        time_of_day: Optional[float] = None,
        pause_cap_s: Optional[float] = None,
    ) -> dict:
        model = self._load_tpp_model(checkpoint)
        if model is None:
            print("[StealthScheduler] TPP checkpoint not found — falling back to static")
            return self._schedule_static(media_ids, base_delay)

        hour = (
            float(self._tpp_rng.integers(0, 24))
            if time_of_day is None
            else float(time_of_day)
        )
        cooldowns = {
            c: CHANNEL_COOLDOWNS.get(c, 6.0) for c in range(self.num_channels)
        }
        delays, channels = model.generate_routed_schedule(
            num_items=len(media_ids),
            time_of_day=hour,
            cooldowns=cooldowns,
            pause_cap_s=pause_cap_s,
            rng=self._tpp_rng,
            device=self.device,
        )
        return {
            "items": media_ids,
            "delays": [float(d) for d in delays],
            "channels": [int(c) for c in channels],
            "mode_used": "tpp",
        }

    def _load_tpp_ppo_agent(self, checkpoint: Optional[Path]):
        if checkpoint is None or not Path(checkpoint).is_file():
            return None
        resolved = Path(checkpoint).resolve()
        if (
            self._tpp_ppo_agent is not None
            and self._tpp_ppo_checkpoint_path == resolved
        ):
            return self._tpp_ppo_agent
        from src.stealth.rl.tpp_ppo import PriorGuidedPPOAgent

        self._tpp_ppo_agent = PriorGuidedPPOAgent.load_checkpoint(
            resolved, device=self.device
        )
        self._tpp_ppo_checkpoint_path = resolved
        return self._tpp_ppo_agent

    def _schedule_tpp_ppo(
        self,
        media_ids: list[str],
        base_delay: float,
        checkpoint: Optional[Path],
        *,
        time_of_day: Optional[float] = None,
        pause_cap_s: Optional[float] = None,
    ) -> dict:
        agent = self._load_tpp_ppo_agent(checkpoint)
        if agent is None:
            print("[StealthScheduler] TPP-PPO checkpoint not found — falling back to static")
            return self._schedule_static(media_ids, base_delay)

        hour = (
            float(self._tpp_rng.integers(0, 24))
            if time_of_day is None
            else float(time_of_day)
        )
        delays, channels = agent.generate_schedule(
            num_items=len(media_ids),
            time_of_day=hour,
            pause_cap_s=pause_cap_s,
            rng=self._tpp_rng,
        )
        return {
            "items": media_ids,
            "delays": [float(d) for d in delays],
            "channels": [int(c) for c in channels],
            "mode_used": "tpp_ppo",
        }

    # ------------------------------------------------------------------
    # static (NoiseController) fallback
    # ------------------------------------------------------------------

    def _schedule_static(self, media_ids: list[str], base_delay: float) -> dict:
        profile_kwargs = ACTIVITY_PROFILES.get(
            self.profile, ACTIVITY_PROFILES["casual"]
        )
        noise = NoiseController(seed=self.seed, **profile_kwargs)

        base_delays = [int(base_delay)] * len(media_ids)
        items, delays = noise.apply(media_ids, base_delays)
        channels = [i % self.num_channels for i in range(len(items))]

        return {
            "items": items,
            "delays": [float(d) for d in delays],
            "channels": channels,
            "mode_used": "static",
        }

    # ------------------------------------------------------------------
    # GAN-based scheduling
    # ------------------------------------------------------------------

    def _route_gan_with_cooldowns(
        self,
        delays: list[float],
        channel_logits: np.ndarray,
        *,
        sample_channels: bool,
    ) -> tuple[list[float], list[int]]:
        """Route GAN timings onto legal channel landings, extending waits as needed."""
        routed_delays = [float(delay) for delay in delays]
        last_landing = {channel: -100.0 for channel in range(self.num_channels)}
        current_landing = 0.0
        routed_channels: list[int] = []

        for index, logits in enumerate(channel_logits):
            current_landing += routed_delays[index]
            available = [
                channel
                for channel in range(self.num_channels)
                if current_landing - last_landing[channel]
                >= CHANNEL_COOLDOWNS.get(channel, 6.0)
            ]
            if available:
                row = np.asarray(logits, dtype=np.float64)
                available_logits = np.asarray([row[channel] for channel in available])
                weights = np.exp(available_logits - np.max(available_logits))
                available_probabilities = weights / weights.sum()
                if sample_channels:
                    chosen = int(np.random.choice(available, p=available_probabilities))
                else:
                    chosen = available[int(np.argmax(available_probabilities))]
            else:
                chosen = min(
                    range(self.num_channels),
                    key=lambda channel: last_landing[channel]
                    + CHANNEL_COOLDOWNS.get(channel, 6.0),
                )
                wait = (
                    last_landing[chosen]
                    + CHANNEL_COOLDOWNS.get(chosen, 6.0)
                    - current_landing
                )
                routed_delays[index] += wait
                current_landing += wait

            last_landing[chosen] = current_landing
            routed_channels.append(chosen)

        return routed_delays, routed_channels

    def _load_generator(self, checkpoint: Optional[Path]):
        if self._generator is not None:
            return True

        if checkpoint is None or not Path(checkpoint).exists():
            return False

        from src.stealth.gan.generator import TemporalPatternGenerator

        # Compatibility shim for Python 3.12 + PyTorch 2.14: pathlib._local module
        # missing when unpickling checkpoints saved with older Python/PyTorch.
        import sys
        import pathlib
        sys.modules.setdefault("pathlib._local", pathlib)

        ckpt = load_trained_checkpoint(checkpoint, map_location=self.device,)
        # Prefer dimensions stored in the checkpoint; fall back to the
        # architecture defaults they were trained with.
        config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        if not isinstance(config, dict):  # TrainingConfig dataclass from the trainer
            config = {
                "latent_dim": getattr(config, "latent_dim", 128),
                "hidden_dim": getattr(config, "hidden_dim", 256),
                "max_sequence_length": 100,
            }
        self._generator = TemporalPatternGenerator(
            latent_dim=config.get("latent_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            num_channels=self.num_channels,
            max_sequence_length=config.get("max_sequence_length", 100),
        )
        self._generator.load_state_dict(ckpt["generator_state"])
        self._generator.to(self.device)
        self._generator.eval()
        return True

    def _schedule_gan(
        self,
        media_ids: list[str],
        base_delay: float,
        checkpoint: Optional[Path],
        time_of_day: Optional[float] = None,
    ) -> dict:
        if not self._load_generator(checkpoint):
            print(
                "[StealthScheduler] GAN checkpoint not found — falling back to static"
            )
            return self._schedule_static(media_ids, base_delay)


        seq_len = len(media_ids)
        hour = (
            float(np.random.randint(0, 24)) if time_of_day is None else float(time_of_day)
        )
        tod = torch.tensor([hour], device=self.device)

        with torch.no_grad():
            schedule = self._generator.generate(
                batch_size=1,
                sequence_length=seq_len,
                time_of_day=tod,
                device=self.device,
            )

        raw_delays = schedule.delays[0].cpu().numpy()
        # Use the same canonical mapping as the hybrid path and the C3
        # benchmark, so the measured GAN row is the deployed GAN row.
        mapped_delays = canonical_delay_mapping(raw_delays).tolist()
        channel_logits = schedule.channel_logits[0].cpu().numpy()
        delays, channels = self._route_gan_with_cooldowns(
            mapped_delays, channel_logits, sample_channels=True
        )

        return {
            "items": media_ids,
            "delays": delays,
            "channels": channels,
            "mode_used": "gan",
            # Pre-mapping generator output, for diagnostics that need to
            # report the generator's own units (e.g. the C3 diurnal check).
            "raw_delays": [float(d) for d in raw_delays.tolist()],
        }

    # ------------------------------------------------------------------
    # Hybrid WGAN + Action-Masked Channel Scheduling
    # ------------------------------------------------------------------

    def _schedule_hybrid(
        self,
        media_ids: list[str],
        base_delay: float,
        checkpoint: Optional[Path],
        time_of_day: Optional[float] = None,
    ) -> dict:
        """
        Hybrid Mode: Combines WGAN generative temporal pauses with closed-loop
        action-masked channel routing so every exported landing respects the
        per-channel cooldowns (same table the benchmark measures).
        """
        if not self._load_generator(checkpoint):
            print("[StealthScheduler] GAN checkpoint not found for hybrid — falling back to static")
            return self._schedule_static(media_ids, base_delay)

        seq_len = len(media_ids)
        hour = (
            float(np.random.randint(0, 24)) if time_of_day is None else float(time_of_day)
        )
        tod = torch.tensor([hour], device=self.device)

        with torch.no_grad():
            schedule = self._generator.generate(
                batch_size=1,
                sequence_length=seq_len,
                time_of_day=tod,
                device=self.device,
            )

        raw_delays = schedule.delays[0].cpu().numpy()
        mapped_delays = canonical_delay_mapping(raw_delays).tolist()
        channel_logits = schedule.channel_logits[0].cpu().numpy()
        delays, routed_channels = self._route_gan_with_cooldowns(
            mapped_delays, channel_logits, sample_channels=False
        )

        return {
            "items": media_ids,
            "delays": [float(d) for d in delays],
            "channels": routed_channels,
            "mode_used": "hybrid",
        }

    # ------------------------------------------------------------------
    # RL-based scheduling
    # ------------------------------------------------------------------

    def _load_rl_agent(self, checkpoint: Optional[Path]):
        if self._rl_agent is not None:
            return True

        if checkpoint is None or not Path(checkpoint).exists():
            return False

        from src.stealth.rl.agent import PPOAgent, PPOConfig
        from src.stealth.rl.environment import StealthEnvironment
        from src.analysis.adversarial.warden import DeepPacketInspectionWarden
        from src.stealth.gan.trainer import apply_critic_config

        # The RL policy's learned behavior does not depend on this Warden
        # instance at inference time (the policy weights already encode
        # what it learned against the Phase-A Warden during training).
        # Still, env.step() computes a reward internally on every call
        # (discarded here, but used by env.render()/get_warden_score() for
        # diagnostics) — load the real trained Warden when available so
        # those diagnostics aren't computed against random noise.
        warden = DeepPacketInspectionWarden(num_channels=self.num_channels)
        if DEFAULT_GAN_CHECKPOINT.exists():
            try:
                # Compatibility shim for Python 3.12 + PyTorch 2.14
                import sys
                import pathlib
                sys.modules.setdefault("pathlib._local", pathlib)
                gan_ckpt = load_trained_checkpoint(
                    DEFAULT_GAN_CHECKPOINT, map_location=self.device,)
                warden.load_state_dict(gan_ckpt["warden_state"])
                apply_critic_config(warden, gan_ckpt)
            except Exception as exc:  # noqa: BLE001 - diagnostics-only fallback
                print(
                    f"[StealthScheduler] Could not load trained Warden for RL "
                    f"diagnostics ({exc}); using an untrained Warden instead. "
                    "This does not affect the RL policy's own behavior."
                )
        warden.eval()

        env = StealthEnvironment(num_channels=self.num_channels, warden=warden)
        config = PPOConfig(state_dim=env.state_dim, device=self.device)
        self._rl_agent = PPOAgent(env, config)
        self._rl_agent.load(checkpoint)
        self._rl_agent.actor_critic.eval()
        return True

    def _schedule_rl(
        self, media_ids: list[str], base_delay: float, checkpoint: Optional[Path]
    ) -> dict:
        if not self._load_rl_agent(checkpoint):
            print("[StealthScheduler] RL checkpoint not found — falling back to static")
            return self._schedule_static(media_ids, base_delay)

        # Drop any stale rollout data from a previous schedule() call —
        # this scheduler is not re-entrant, see the class docstring.
        self._rl_agent.buffer.clear()

        env = self._rl_agent.env
        state = env.reset(media_ids)
        delays, channels = [], []

        # Landing-clock bookkeeping for the exported schedule, in the same
        # terms the Component 3 benchmark measures: cumsum(delays)[k] is
        # item k's landing time, prev_landing is the landing time of the
        # item just placed, and last_landing[] is the last landing per
        # channel (same -100 initial offset as the hybrid router).
        prev_landing = 0.0
        last_landing = {c: -100.0 for c in range(self.num_channels)}

        # Drive the policy with the closed-loop action mask (the same mask
        # the training rollout used), so no transmission targets a channel
        # that is still cooling down; env.step can end the episode only
        # with the timeout or an empty queue, and a cooldown pick inside
        # step() is a rate-limit back-off extension, never a rejection —
        # the environment emits no "rate_limit_violation" outcome at all.
        max_steps = len(media_ids) * 3
        for _ in range(max_steps):
            if len(delays) >= len(media_ids):
                break
            mask = env.get_action_mask()
            action, _, _ = self._rl_agent.select_action(state, channel_mask=mask)
            next_state, _, done, info = env.step(action)

            if info.get("reason") == "timeout":
                # The episode clock expired before this item could be sent:
                # the env returns before dequeuing media and the info dict
                # carries no delay, so nothing reached the wire for this
                # step. Export only confirmed transmissions; the tail is
                # filled below by cooldown-aware routing rather than by a
                # phantom delay.
                break

            # The delay is the one the environment actually used, after any
            # rate-limit back-off. The agent clamps its action into the
            # deployment window before the environment scores it, and the
            # environment reports the delay it landed on, so the delay that
            # was rewarded is the delay that goes on the wire. The previous
            # transform, `clip(raw * 0.35, 7, 20)`, applied here instead
            # meant the agent optimized a delay sequence that never reached
            # the network, and its 7 second floor collapsed 98.3 percent of
            # the agent's choices onto a single value, so the transmitted
            # traffic was a constant clock. See src/stealth/rl/pacing.py.
            delay_used = float(info.get("delay", action["delay"]))
            chosen = int(action["channel"]) % self.num_channels
            delays.append(delay_used)
            channels.append(chosen)
            last_landing[chosen] = prev_landing + delay_used
            prev_landing = last_landing[chosen]
            state = next_state
            if done:
                break

        # Fill if RL ended early. The only current exit is the episode
        # timeout, where the env sent nothing on the step that ended the
        # episode. The remaining items are filled by per-channel cooldown
        # routing, the same earliest-free-channel rule the hybrid router
        # uses: each fill item lands at the earliest moment that respects
        # both the pacing floor (RL_DELAY_MIN) and the cooldown clock of
        # its channel, measured on the landing times the benchmark scores.
        # The old fill appended RL_DELAY_MIN on channel 0 regardless, so
        # a tail of more than one item violated channel 0's cooldown.
        while len(delays) < len(media_ids):
            chosen = min(
                range(self.num_channels),
                key=lambda c: last_landing[c] + CHANNEL_COOLDOWNS[c],
            )
            gap = max(
                RL_DELAY_MIN,
                last_landing[chosen] + CHANNEL_COOLDOWNS[chosen] - prev_landing,
            )
            delays.append(gap)
            channels.append(chosen)
            last_landing[chosen] = prev_landing + gap
            prev_landing += gap

        return {
            "items": media_ids,
            "delays": delays,
            "channels": channels,
            "mode_used": "rl",
        }
