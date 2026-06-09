# src/stealth/stealth_scheduler.py
"""
Stealth Scheduler — Integration Bridge.

Connects trained GAN Generator / RL Agent outputs to the distribution
pipeline (Scheduler + Dispatcher).  If no trained model is available
the system falls back to static NoiseController scheduling.
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Literal, Optional

from src.distribution.noise import NoiseController
from src.distribution.profiles import ACTIVITY_PROFILES


class StealthScheduler:
    """
    Unified scheduler that can operate in three modes:

    * **static**  – handcrafted NoiseController (always available)
    * **gan**     – TemporalPatternGenerator (needs trained checkpoint)
    * **rl**      – PPOAgent policy (needs trained checkpoint)

    If a checkpoint is requested but missing the scheduler automatically
    falls back to *static* mode and logs a warning.
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
        mode: Literal["static", "gan", "rl"] = "static",
        base_delay: float = 3.0,
        gan_checkpoint: Optional[Path] = None,
        rl_checkpoint: Optional[Path] = None,
    ) -> dict:
        """
        Produce a timed schedule for *media_ids*.

        Returns
        -------
        dict with keys:
            items    – list[str]  media IDs after noise filtering
            delays   – list[float]  inter-item delays in seconds
            channels – list[int]  channel index per item
            mode_used – str  actual mode that ran (may differ from *mode* on fallback)
        """
        if mode == "gan":
            return self._schedule_gan(media_ids, base_delay, gan_checkpoint)
        elif mode == "rl":
            return self._schedule_rl(media_ids, base_delay, rl_checkpoint)
        else:
            return self._schedule_static(media_ids, base_delay)

    # ------------------------------------------------------------------
    # static (NoiseController) fallback
    # ------------------------------------------------------------------

    def _schedule_static(self, media_ids: list[str], base_delay: float) -> dict:
        profile_kwargs = ACTIVITY_PROFILES.get(self.profile, ACTIVITY_PROFILES["casual"])
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

        self._generator = TemporalPatternGenerator(
            latent_dim=128, hidden_dim=256,
            num_channels=self.num_channels, max_sequence_length=100,
        )
        ckpt = torch.load(checkpoint, map_location=self.device)
        self._generator.load_state_dict(ckpt["generator_state"])
        self._generator.to(self.device)
        self._generator.eval()
        return True

    def _schedule_gan(self, media_ids: list[str], base_delay: float,
                      checkpoint: Optional[Path]) -> dict:
        if not self._load_generator(checkpoint):
            print("[StealthScheduler] GAN checkpoint not found — falling back to static")
            return self._schedule_static(media_ids, base_delay)

        import torch
        seq_len = len(media_ids)
        time_of_day = torch.tensor([float(np.random.randint(0, 24))], device=self.device)

        with torch.no_grad():
            schedule = self._generator.generate(
                batch_size=1, sequence_length=seq_len, time_of_day=time_of_day,
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
    # RL-based scheduling
    # ------------------------------------------------------------------

    def _load_rl_agent(self, checkpoint: Optional[Path]):
        if self._rl_agent is not None:
            return True

        if checkpoint is None or not Path(checkpoint).exists():
            return False

        from src.stealth.rl.agent import PPOAgent, PPOConfig, ActorCritic
        from src.stealth.rl.environment import StealthEnvironment
        from src.analysis.adversarial.warden import DeepPacketInspectionWarden

        warden = DeepPacketInspectionWarden(num_channels=self.num_channels)
        warden.eval()

        env = StealthEnvironment(num_channels=self.num_channels, warden=warden)
        config = PPOConfig(state_dim=env.state_dim, device=self.device)
        self._rl_agent = PPOAgent(env, config)
        self._rl_agent.load(checkpoint)
        self._rl_agent.actor_critic.eval()
        return True

    def _schedule_rl(self, media_ids: list[str], base_delay: float,
                     checkpoint: Optional[Path]) -> dict:
        if not self._load_rl_agent(checkpoint):
            print("[StealthScheduler] RL checkpoint not found — falling back to static")
            return self._schedule_static(media_ids, base_delay)

        env = self._rl_agent.env
        state = env.reset(media_ids)
        delays, channels = [], []

        for _ in range(len(media_ids)):
            action, _, _ = self._rl_agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            delays.append(max(0.5, float(action["delay"])))
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
