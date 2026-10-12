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
import numpy as np
from pathlib import Path
from typing import Literal, Optional

from src.distribution.noise import NoiseController
from src.distribution.profiles import ACTIVITY_PROFILES

# Anchored to the project root so schedules work regardless of CWD.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_GAN_CHECKPOINT = _PROJECT_ROOT / "storage" / "models" / "gan_generator.pt"
DEFAULT_RL_CHECKPOINT = _PROJECT_ROOT / "storage" / "models" / "rl_agent.pt"


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

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        media_ids: list[str],
        mode: Literal["static", "gan", "rl", "hybrid", "auto"] = "static",
        base_delay: float = 3.0,
        gan_checkpoint: Optional[Path] = None,
        rl_checkpoint: Optional[Path] = None,
    ) -> dict:
        """
        Produce a timed schedule for *media_ids*.

        mode="auto" runs the rl -> gan -> static cascade, matching the
        documented behaviour in scripts/runtime/run_sender.py.
        mode="hybrid" combines WGAN timing pauses with closed-loop channel
        cooldown tracking to guarantee zero rate-limit violations.

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
            return self._schedule_unlocked(media_ids, mode, base_delay, gan_checkpoint, rl_checkpoint)

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
        mode: Literal["static", "gan", "rl", "hybrid", "auto"],
        base_delay: float,
        gan_checkpoint: Optional[Path],
        rl_checkpoint: Optional[Path],
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
                )
            if schedule["mode_used"] not in ("rl", "gan"):
                schedule = self._schedule_static(media_ids, base_delay)
            return self._validate_schedule(schedule, media_ids)

        if mode == "hybrid":
            checkpoint = self._resolve_checkpoint(
                gan_checkpoint, DEFAULT_GAN_CHECKPOINT
            )
            schedule = self._schedule_hybrid(media_ids, base_delay, checkpoint)
        elif mode == "gan":
            checkpoint = self._resolve_checkpoint(
                gan_checkpoint, DEFAULT_GAN_CHECKPOINT
            )
            schedule = self._schedule_gan(media_ids, base_delay, checkpoint)
        elif mode == "rl":
            checkpoint = self._resolve_checkpoint(rl_checkpoint, DEFAULT_RL_CHECKPOINT)
            schedule = self._schedule_rl(media_ids, base_delay, checkpoint)
        else:
            schedule = self._schedule_static(media_ids, base_delay)

        return self._validate_schedule(schedule, media_ids)

    @staticmethod
    def _resolve_checkpoint(explicit: Optional[Path], default: Path) -> Optional[Path]:
        """Prefer an explicit path; otherwise use the project-root default if present."""
        if explicit is not None:
            return Path(explicit)
        return default if default.exists() else None

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

    def _load_generator(self, checkpoint: Optional[Path]):
        if self._generator is not None:
            return True

        if checkpoint is None or not Path(checkpoint).exists():
            return False

        from src.stealth.gan.generator import TemporalPatternGenerator

        # Compatibility shim for Python 3.12 + PyTorch 2.14: pathlib._local module
        # missing when unpickling checkpoints saved with older Python/PyTorch.
        import sys, pathlib
        sys.modules.setdefault("pathlib._local", pathlib)

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
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
        self, media_ids: list[str], base_delay: float, checkpoint: Optional[Path]
    ) -> dict:
        if not self._load_generator(checkpoint):
            print(
                "[StealthScheduler] GAN checkpoint not found — falling back to static"
            )
            return self._schedule_static(media_ids, base_delay)

        import torch

        seq_len = len(media_ids)
        time_of_day = torch.tensor(
            [float(np.random.randint(0, 24))], device=self.device
        )

        with torch.no_grad():
            schedule = self._generator.generate(
                batch_size=1,
                sequence_length=seq_len,
                time_of_day=time_of_day,
                device=self.device,
            )

        delays = schedule.delays[0].cpu().tolist()
        channels = schedule.sample_channels()[0].cpu().tolist()

        return {
            "items": media_ids,
            "delays": delays,
            "channels": channels,
            "mode_used": "gan",
        }

    # ------------------------------------------------------------------
    # Hybrid WGAN + Action-Masked Channel Scheduling
    # ------------------------------------------------------------------

    def _schedule_hybrid(
        self, media_ids: list[str], base_delay: float, checkpoint: Optional[Path]
    ) -> dict:
        """
        Hybrid Mode: Combines WGAN generative temporal pauses with closed-loop
        action-masked channel routing to guarantee zero rate-limit cooldown violations.
        """
        if not self._load_generator(checkpoint):
            print("[StealthScheduler] GAN checkpoint not found for hybrid — falling back to static")
            return self._schedule_static(media_ids, base_delay)

        import torch
        seq_len = len(media_ids)
        time_of_day = torch.tensor([float(np.random.randint(0, 24))], device=self.device)

        with torch.no_grad():
            schedule = self._generator.generate(
                batch_size=1,
                sequence_length=seq_len,
                time_of_day=time_of_day,
                device=self.device,
            )

        raw_delays = schedule.delays[0].cpu().numpy()
        delays = np.clip((raw_delays - 0.5) * 1500.0 + 1.5, 0.5, 30.0).tolist()
        channel_logits = schedule.channel_logits[0].cpu().numpy()

        # Closed-loop channel cooldown routing
        # Enterprise cooldowns: ch0 = 6.0s (10/min), ch1 = 12.0s (5/min), ch2 = 4.0s (15/min)
        cooldowns = {0: 6.0, 1: 12.0, 2: 4.0}
        t_last = {c: -100.0 for c in range(self.num_channels)}
        t_current = 0.0
        routed_channels = []

        for i in range(seq_len):
            t_current += delays[i]
            avail = [c for c in range(self.num_channels) if (t_current - t_last[c]) >= cooldowns.get(c, 6.0)]
            if avail:
                probs = np.exp(channel_logits[i]) / np.sum(np.exp(channel_logits[i]))
                avail_probs = [probs[c] for c in avail]
                chosen_c = avail[int(np.argmax(avail_probs))]
            else:
                earliest_c = min(range(self.num_channels), key=lambda c: t_last[c] + cooldowns.get(c, 6.0))
                wait = (t_last[earliest_c] + cooldowns.get(earliest_c, 6.0)) - t_current
                delays[i] += wait
                t_current += wait
                chosen_c = earliest_c

            t_last[chosen_c] = t_current
            routed_channels.append(chosen_c)

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
                import sys, pathlib
                sys.modules.setdefault("pathlib._local", pathlib)
                gan_ckpt = torch.load(
                    DEFAULT_GAN_CHECKPOINT, map_location=self.device, weights_only=False
                )
                warden.load_state_dict(gan_ckpt["warden_state"])
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

        for _ in range(len(media_ids)):
            # Pass the action mask so the policy doesn't pick a
            # rate-limit-cooling channel at inference (it already does
            # during training via collect_rollout). This also means
            # "rate_limit_violation" steps should no longer occur here.
            mask = env.get_action_mask()
            action, _, _ = self._rl_agent.select_action(state, channel_mask=mask)
            next_state, _, done, info = env.step(action)

            if info.get("reason") == "rate_limit_violation":
                # Should be rare now that the mask is applied, but if the
                # policy still picks a masked channel (e.g. all channels
                # cooling down), don't let a phantom step leak into the
                # exported schedule — retry is not possible without
                # consuming the queue, so stop here and pad below.
                state = next_state
                continue

            raw_d = float(action["delay"])
            # Calibrate delay pacing to channel cooldown constraints (7s to 20s)
            delays.append(float(np.clip(raw_d * 0.35, 7.0, 20.0)))
            channels.append(int(action["channel"]) % self.num_channels)
            state = next_state
            if done:
                break

        # Pad if RL ended early
        while len(delays) < len(media_ids):
            delays.append(base_delay)
            channels.append(0)

        return {
            "items": media_ids,
            "delays": delays,
            "channels": channels,
            "mode_used": "rl",
        }
