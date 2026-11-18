"""Prior-Guided KL-Regularized PPO Scheduler over a Frozen Autoregressive TPP.

Instead of inventing human timing from scratch with a hard-clamped unimodal
Gaussian, this controller sits on top of a trained TemporalPointProcessGenerator:

1. At each step t, the frozen TPP proposes a K-component Log-Normal mixture
   for log(delay_t) and conditional channel logits given recurrent history
   and hour-of-day.
2. The PPO actor observes closed-loop operational state (channel cooldown
   headroom, 5-second burst density, queue backlog, recent CV/entropy, and
   TPP prior summaries) and outputs bounded residual adjustments to:
     - the mixture component logits,
     - the chosen component's log-delay mean and scale,
     - the cooldown-masked channel logits.
3. Zero-initialized residual heads start the policy exactly on the learned
   human TPP manifold, and a per-step KL penalty to the TPP prior prevents
   reward hacking against any single detector so gains transfer to unseen
   held-out evaluators.
4. Every step enforces float64 per-channel cooldowns and feeds the realized
   (delay_t, channel_t) back into the TPP recurrent state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from src.analysis.adversarial.warden import DETECTOR_SEQ_LEN
from src.stealth.tpp.model import TemporalPointProcessGenerator
from src.utils.checkpoint_io import load_trained_checkpoint


@dataclass
class PriorGuidedPPOConfig:
    num_channels: int = 3
    num_components: int = 8
    hidden_dim: int = 128
    max_mix_delta: float = 1.0
    max_log_shift: float = 0.20
    max_log_scale_delta: float = 0.15
    max_ch_delta: float = 1.2
    pause_cap_s: Optional[float] = 60.0
    lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    kl_beta: float = 0.60
    kl_loss_coef: float = 0.60
    lambda_stealth: float = 4.0
    burst_penalty: float = 1.5
    latency_budget_s: Optional[float] = None
    latency_penalty: float = 0.25
    ppo_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PriorGuidedPPOConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def compute_controller_state_dim(num_channels: int, num_components: int) -> int:
    # queue_frac (1) + hour sin/cos (2) + per-channel [headroom, usage] (2*C)
    # + recent stats [log_mean, log_std, cv, burst5_norm, elapsed_norm] (5)
    # + TPP prior summary [mix_probs (K), prior_log_mean (1), prior_log_std (1)] (K+2)
    return 1 + 2 + 2 * num_channels + 5 + num_components + 2


def build_controller_state(
    *,
    remaining: int,
    total_items: int,
    current_hour: float,
    current_landing: float,
    last_landing: Dict[int, float],
    channel_counts: List[int],
    cooldowns: Dict[int, float],
    delays_so_far: List[float],
    timestamps_so_far: List[float],
    prior_mix_logits: torch.Tensor,
    prior_means: torch.Tensor,
    prior_scales: torch.Tensor,
) -> np.ndarray:
    num_channels = len(channel_counts)
    step = len(delays_so_far)

    feats: list[float] = []
    feats.append(float(remaining) / max(total_items, 1))

    angle = 2.0 * math.pi * (current_hour % 24.0) / 24.0
    feats.append(math.sin(angle))
    feats.append(math.cos(angle))

    for c in range(num_channels):
        cd = max(float(cooldowns.get(c, 6.0)), 1e-3)
        headroom = (current_landing - last_landing[c]) / cd
        feats.append(float(np.clip(headroom, 0.0, 2.5)))
        feats.append(float(channel_counts[c]) / max(step, 1))

    if step > 0:
        arr = np.asarray(delays_so_far[-20:], dtype=np.float64)
        mu = float(arr.mean())
        sd = float(arr.std())
        cv = sd / (mu + 1e-8)
        ts = np.asarray(timestamps_so_far, dtype=np.float64)
        burst5 = float(np.sum(ts > (current_landing - 5.0)))
        feats.append(math.log1p(max(mu, 0.0)) / 5.0)
        feats.append(math.log1p(max(sd, 0.0)) / 5.0)
        feats.append(float(np.clip(cv, 0.0, 3.0)) / 3.0)
        feats.append(float(np.clip(burst5 / 6.0, 0.0, 2.0)))
        feats.append(float(np.clip((current_landing / 60.0) / 20.0, 0.0, 2.0)))
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    with torch.no_grad():
        probs = F.softmax(prior_mix_logits, dim=-1).cpu().numpy()
        means_np = prior_means.cpu().numpy()
        scales_np = prior_scales.cpu().numpy()
        prior_mean = float(np.sum(probs * means_np))
        prior_var = float(np.sum(probs * (scales_np**2 + (means_np - prior_mean) ** 2)))
        feats.extend(probs.tolist())
        feats.append(prior_mean / 5.0)
        feats.append(math.sqrt(max(prior_var, 1e-6)) / 3.0)

    return np.asarray(feats, dtype=np.float32)


class PriorGuidedActorCritic(nn.Module):
    """Residual Actor-Critic anchored to a frozen TPP prior."""

    def __init__(self, config: PriorGuidedPPOConfig):
        super().__init__()
        self.cfg = config
        state_dim = compute_controller_state_dim(
            config.num_channels, config.num_components
        )
        self.state_dim = state_dim

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Residual heads — zero-initialized so policy begins identically to TPP.
        self.delta_mix_head = nn.Linear(config.hidden_dim, config.num_components)
        self.delta_delay_head = nn.Linear(config.hidden_dim, 2 * config.num_components)
        self.delta_ch_head = nn.Linear(config.hidden_dim, config.num_channels)

        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        for head in (self.delta_mix_head, self.delta_delay_head, self.delta_ch_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(state)
        delta_mix = self.cfg.max_mix_delta * torch.tanh(self.delta_mix_head(h))
        raw_d = self.delta_delay_head(h)
        raw_shift, raw_scale = torch.chunk(raw_d, 2, dim=-1)
        delta_mu = self.cfg.max_log_shift * torch.tanh(raw_shift)
        scale_mult = torch.exp(self.cfg.max_log_scale_delta * torch.tanh(raw_scale))
        delta_ch = self.cfg.max_ch_delta * torch.tanh(self.delta_ch_head(h))
        value = self.value_head(h).squeeze(-1)
        return delta_mix, delta_mu, scale_mult, delta_ch, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        prior_mix_logits: torch.Tensor,
        prior_means: torch.Tensor,
        prior_scales: torch.Tensor,
        prior_ch_logits: torch.Tensor,
        ch_masks: torch.Tensor,
        comp_actions: torch.Tensor,
        log_delay_actions: torch.Tensor,
        ch_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (log_prob, value, entropy, kl_to_prior) for a batch of steps."""
        delta_mix, delta_mu, scale_mult, delta_ch, values = self.forward(states)

        # 1. Mixture component distribution
        pol_mix_logits = prior_mix_logits + delta_mix
        pol_mix_dist = Categorical(logits=pol_mix_logits)
        prior_mix_dist = Categorical(logits=prior_mix_logits)
        lp_comp = pol_mix_dist.log_prob(comp_actions)
        ent_comp = pol_mix_dist.entropy()
        kl_mix = torch.distributions.kl_divergence(pol_mix_dist, prior_mix_dist)

        # 2. Selected component log-delay Gaussian
        idx = comp_actions.unsqueeze(-1)
        p_mu = prior_means.gather(-1, idx).squeeze(-1)
        p_sd = prior_scales.gather(-1, idx).squeeze(-1)
        d_mu = delta_mu.gather(-1, idx).squeeze(-1)
        s_mult = scale_mult.gather(-1, idx).squeeze(-1)

        pol_delay_dist = Normal(p_mu + d_mu, (p_sd * s_mult).clamp(min=0.05))
        prior_delay_dist = Normal(p_mu, p_sd.clamp(min=0.05))
        lp_delay = pol_delay_dist.log_prob(log_delay_actions)
        ent_delay = pol_delay_dist.entropy()
        kl_delay = torch.distributions.kl_divergence(pol_delay_dist, prior_delay_dist)

        # 3. Cooldown-masked channel distribution
        mask_bias = (1.0 - ch_masks) * -1e9
        pol_ch_dist = Categorical(logits=prior_ch_logits + delta_ch + mask_bias)
        prior_ch_dist = Categorical(logits=prior_ch_logits + mask_bias)
        lp_ch = pol_ch_dist.log_prob(ch_actions)
        ent_ch = pol_ch_dist.entropy()
        kl_ch = torch.distributions.kl_divergence(pol_ch_dist, prior_ch_dist)

        total_lp = lp_comp + lp_delay + lp_ch
        total_ent = ent_comp + ent_delay + ent_ch
        total_kl = kl_mix + kl_delay + kl_ch
        return total_lp, values, total_ent, total_kl


def _score_sequence_with_warden(
    warden: nn.Module,
    delays: Sequence[float],
    channels: Sequence[int],
    device: torch.device,
) -> tuple[float, float]:
    """Score a trajectory using the exact DETECTOR_SEQ_LEN masked convention."""
    n = min(len(delays), DETECTOR_SEQ_LEN)
    if n == 0:
        return 0.0, 0.0
    pad_d = np.zeros(DETECTOR_SEQ_LEN, dtype=np.float32)
    pad_c = np.zeros(DETECTOR_SEQ_LEN, dtype=np.int64)
    mask = np.zeros(DETECTOR_SEQ_LEN, dtype=np.float32)
    pad_d[:n] = np.asarray(delays[-n:], dtype=np.float32)
    pad_c[:n] = np.asarray(channels[-n:], dtype=np.int64)
    mask[:n] = 1.0

    with torch.no_grad():
        verdict = warden(
            torch.from_numpy(pad_d).unsqueeze(0).to(device),
            torch.from_numpy(pad_c).unsqueeze(0).to(device),
            mask=torch.from_numpy(mask).unsqueeze(0).to(device),
        )
        bot_prob = float(verdict.bot_probability[0].item())
        anom_mean = float(verdict.anomaly_scores[0, :n].mean().item())
    return bot_prob, anom_mean


class PriorGuidedPPOAgent:
    """Closed-loop KL-regularized PPO controller anchored to a frozen TPP."""

    def __init__(
        self,
        tpp: TemporalPointProcessGenerator,
        config: Optional[PriorGuidedPPOConfig] = None,
        cooldowns: Optional[Dict[int, float]] = None,
    ):
        self.tpp = tpp
        self.tpp.eval()
        for p in self.tpp.parameters():
            p.requires_grad_(False)

        self.cfg = config or PriorGuidedPPOConfig(
            num_channels=tpp.num_channels,
            num_components=tpp.num_components,
        )
        self.device = torch.device(self.cfg.device)
        self.tpp.to(self.device)
        self.cooldowns = (
            {0: 6.0, 1: 12.0, 2: 4.0}
            if cooldowns is None
            else {int(k): float(v) for k, v in cooldowns.items()}
        )

        self.actor_critic = PriorGuidedActorCritic(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=self.cfg.lr
        )

    @torch.no_grad()
    def generate_schedule(
        self,
        num_items: int,
        time_of_day: float = 14.0,
        pause_cap_s: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[list[float], list[int]]:
        """Run the closed-loop Prior-Guided PPO policy for deployment."""
        self.actor_critic.eval()
        self.tpp.eval()
        cap = (
            float(pause_cap_s)
            if pause_cap_s is not None
            else (
                float(self.cfg.pause_cap_s)
                if self.cfg.pause_cap_s is not None
                else self.tpp.max_delay
            )
        )
        cap = max(self.tpp.min_delay, min(cap, self.tpp.max_delay))

        last_landing = {c: -1000.0 for c in range(self.cfg.num_channels)}
        channel_counts = [0] * self.cfg.num_channels
        current_landing = 0.0
        delays_out: list[float] = []
        channels_out: list[int] = []
        timestamps_out: list[float] = []

        hidden: Optional[torch.Tensor] = None
        prev_delay_t: Optional[torch.Tensor] = None
        prev_ch_t: Optional[torch.Tensor] = None

        for step in range(num_items):
            curr_hour = (float(time_of_day) + current_landing / 3600.0) % 24.0
            hour_t = torch.tensor([curr_hour], dtype=torch.float32, device=self.device)
            mix_logits, means, scales, h_t, hidden = self.tpp.step(
                prev_delay_t, prev_ch_t, hour_t, hidden
            )

            state_np = build_controller_state(
                remaining=num_items - step,
                total_items=num_items,
                current_hour=curr_hour,
                current_landing=current_landing,
                last_landing=last_landing,
                channel_counts=channel_counts,
                cooldowns=self.cooldowns,
                delays_so_far=delays_out,
                timestamps_so_far=timestamps_out,
                prior_mix_logits=mix_logits[0],
                prior_means=means[0],
                prior_scales=scales[0],
            )
            state_t = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
            delta_mix, delta_mu, scale_mult, delta_ch, _ = self.actor_critic(state_t)

            # Sample mixture component
            pol_mix_probs = F.softmax(mix_logits[0] + delta_mix[0], dim=-1).cpu().numpy().astype(np.float64)
            pol_mix_probs /= pol_mix_probs.sum()
            if rng is not None:
                comp = int(rng.choice(self.cfg.num_components, p=pol_mix_probs))
                z = float(rng.standard_normal())
            else:
                comp = int(np.random.choice(self.cfg.num_components, p=pol_mix_probs))
                z = float(np.random.standard_normal())

            mean_k = float((means[0, comp] + delta_mu[0, comp]).item())
            std_k = max(0.05, float((scales[0, comp] * scale_mult[0, comp]).item()))
            log_d = mean_k + std_k * z
            delay = float(np.clip(math.exp(np.clip(log_d, -6.0, 8.0)), self.tpp.min_delay, cap))

            # Channel prediction + cooldown mask
            delay_t = torch.tensor([delay], dtype=torch.float32, device=self.device)
            prior_ch_logits = self.tpp.predict_channel_logits(h_t, delay_t)[0]
            pol_ch_logits = (prior_ch_logits + delta_ch[0]).cpu().numpy().astype(np.float64)

            projected = current_landing + delay
            available = [
                c
                for c in range(self.cfg.num_channels)
                if projected - last_landing[c] >= self.cooldowns.get(c, 6.0)
            ]
            if available:
                avail_logits = np.array([pol_ch_logits[c] for c in available], dtype=np.float64)
                avail_logits -= np.max(avail_logits)
                w = np.exp(avail_logits)
                w /= w.sum()
                chosen = int(rng.choice(available, p=w) if rng is not None else np.random.choice(available, p=w))
            else:
                chosen = min(
                    range(self.cfg.num_channels),
                    key=lambda c: last_landing[c] + self.cooldowns.get(c, 6.0),
                )
                wait = last_landing[chosen] + self.cooldowns.get(chosen, 6.0) - projected
                delay = float(delay + max(0.0, wait))
                projected = current_landing + delay

            current_landing = projected
            last_landing[chosen] = current_landing
            channel_counts[chosen] += 1
            delays_out.append(delay)
            channels_out.append(chosen)
            timestamps_out.append(current_landing)

            prev_delay_t = torch.tensor([delay], dtype=torch.float32, device=self.device)
            prev_ch_t = torch.tensor([chosen], dtype=torch.long, device=self.device)

        return delays_out, channels_out

    def _collect_episode(
        self,
        warden: nn.Module,
        num_items: int,
        start_hour: float,
        score_stride: int = 5,
    ) -> Tuple[dict, dict]:
        """Collect one closed-loop episode with exact aligned 64-step Warden reward."""
        self.actor_critic.eval()
        cap = (
            float(self.cfg.pause_cap_s)
            if self.cfg.pause_cap_s is not None
            else self.tpp.max_delay
        )
        cap = max(self.tpp.min_delay, min(cap, self.tpp.max_delay))

        last_landing = {c: -1000.0 for c in range(self.cfg.num_channels)}
        channel_counts = [0] * self.cfg.num_channels
        current_landing = 0.0
        delays_out: list[float] = []
        channels_out: list[int] = []
        timestamps_out: list[float] = []

        hidden: Optional[torch.Tensor] = None
        prev_delay_t: Optional[torch.Tensor] = None
        prev_ch_t: Optional[torch.Tensor] = None

        traj = {
            "states": [],
            "prior_mix_logits": [],
            "prior_means": [],
            "prior_scales": [],
            "prior_ch_logits": [],
            "ch_masks": [],
            "comp_actions": [],
            "log_delay_actions": [],
            "ch_actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }

        total_kl = 0.0
        total_burst_pen = 0.0

        for step in range(num_items):
            curr_hour = (float(start_hour) + current_landing / 3600.0) % 24.0
            hour_t = torch.tensor([curr_hour], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                mix_logits, means, scales, h_t, hidden = self.tpp.step(
                    prev_delay_t, prev_ch_t, hour_t, hidden
                )

                state_np = build_controller_state(
                    remaining=num_items - step,
                    total_items=num_items,
                    current_hour=curr_hour,
                    current_landing=current_landing,
                    last_landing=last_landing,
                    channel_counts=channel_counts,
                    cooldowns=self.cooldowns,
                    delays_so_far=delays_out,
                    timestamps_so_far=timestamps_out,
                    prior_mix_logits=mix_logits[0],
                    prior_means=means[0],
                    prior_scales=scales[0],
                )
                state_t = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
                delta_mix, delta_mu, scale_mult, delta_ch, value_t = self.actor_critic(state_t)

                pol_mix_dist = Categorical(logits=mix_logits[0] + delta_mix[0])
                prior_mix_dist = Categorical(logits=mix_logits[0])
                comp_t = pol_mix_dist.sample()
                lp_comp = pol_mix_dist.log_prob(comp_t)
                kl_mix = torch.distributions.kl_divergence(pol_mix_dist, prior_mix_dist)

                k = int(comp_t.item())
                p_mu = means[0, k]
                p_sd = scales[0, k].clamp(min=0.05)
                pol_d_dist = Normal(p_mu + delta_mu[0, k], (p_sd * scale_mult[0, k]).clamp(min=0.05))
                prior_d_dist = Normal(p_mu, p_sd)
                log_d_t = pol_d_dist.sample()
                lp_delay = pol_d_dist.log_prob(log_d_t)
                kl_delay = torch.distributions.kl_divergence(pol_d_dist, prior_d_dist)

                delay = float(
                    np.clip(
                        math.exp(np.clip(float(log_d_t.item()), -6.0, 8.0)),
                        self.tpp.min_delay,
                        cap,
                    )
                )

                delay_t = torch.tensor([delay], dtype=torch.float32, device=self.device)
                prior_ch_logits = self.tpp.predict_channel_logits(h_t, delay_t)[0]

                projected = current_landing + delay
                available = [
                    c
                    for c in range(self.cfg.num_channels)
                    if projected - last_landing[c] >= self.cooldowns.get(c, 6.0)
                ]
                ch_mask = torch.zeros(self.cfg.num_channels, dtype=torch.float32, device=self.device)
                if available:
                    for c in available:
                        ch_mask[c] = 1.0
                else:
                    earliest = min(
                        range(self.cfg.num_channels),
                        key=lambda c: last_landing[c] + self.cooldowns.get(c, 6.0),
                    )
                    ch_mask[earliest] = 1.0
                    wait = last_landing[earliest] + self.cooldowns.get(earliest, 6.0) - projected
                    delay = float(delay + max(0.0, wait))
                    projected = current_landing + delay

                mask_bias = (1.0 - ch_mask) * -1e9
                pol_ch_dist = Categorical(logits=prior_ch_logits + delta_ch[0] + mask_bias)
                prior_ch_dist = Categorical(logits=prior_ch_logits + mask_bias)
                ch_t = pol_ch_dist.sample()
                lp_ch = pol_ch_dist.log_prob(ch_t)
                kl_ch = torch.distributions.kl_divergence(pol_ch_dist, prior_ch_dist)

                chosen = int(ch_t.item())
                total_lp = float((lp_comp + lp_delay + lp_ch).item())
                step_kl = float((kl_mix + kl_delay + kl_ch).item())

            # Apply step
            current_landing = projected
            last_landing[chosen] = current_landing
            channel_counts[chosen] += 1
            delays_out.append(delay)
            channels_out.append(chosen)
            timestamps_out.append(current_landing)

            prev_delay_t = torch.tensor([delay], dtype=torch.float32, device=self.device)
            prev_ch_t = torch.tensor([chosen], dtype=torch.long, device=self.device)

            # Burst safety penalty (rule trips at >= 6 packets in any 5s window)
            ts_arr = np.asarray(timestamps_out, dtype=np.float64)
            recent_burst = int(np.sum(ts_arr > (current_landing - 5.0)))
            burst_pen = 0.0
            if recent_burst >= 6:
                burst_pen = 3.0 * self.cfg.burst_penalty
            elif recent_burst == 5:
                burst_pen = 0.6 * self.cfg.burst_penalty

            # Latency budget shaping (optional smooth excess penalty)
            lat_pen = 0.0
            if self.cfg.latency_budget_s is not None and delay > self.cfg.latency_budget_s:
                lat_pen = self.cfg.latency_penalty * (
                    (delay - self.cfg.latency_budget_s) / max(self.cfg.latency_budget_s, 1.0)
                )

            done = step == (num_items - 1)
            stealth_pen = 0.0
            if ((step + 1) >= 10 and (step + 1) % score_stride == 0) or done:
                bot_prob, anom_mean = _score_sequence_with_warden(
                    warden, delays_out, channels_out, self.device
                )
                # Stronger terminal credit on final sequence score
                weight = 3.0 if done else 0.5
                stealth_pen = weight * self.cfg.lambda_stealth * (0.5 * bot_prob + 0.5 * anom_mean)

            # Periodicity guard at episode end
            periodic_pen = 0.0
            if done and len(delays_out) >= 10:
                arr = np.asarray(delays_out, dtype=np.float64)
                cv = float(arr.std() / (arr.mean() + 1e-8))
                if cv < 0.35:
                    periodic_pen = 3.0 * (0.35 - cv)

            reward = (
                -stealth_pen
                - burst_pen
                - lat_pen
                - periodic_pen
                - self.cfg.kl_beta * step_kl
            )
            total_kl += step_kl
            total_burst_pen += burst_pen

            traj["states"].append(state_np)
            traj["prior_mix_logits"].append(mix_logits[0].cpu().numpy())
            traj["prior_means"].append(means[0].cpu().numpy())
            traj["prior_scales"].append(scales[0].cpu().numpy())
            traj["prior_ch_logits"].append(prior_ch_logits.cpu().numpy())
            traj["ch_masks"].append(ch_mask.cpu().numpy())
            traj["comp_actions"].append(k)
            traj["log_delay_actions"].append(float(log_d_t.item()))
            traj["ch_actions"].append(chosen)
            traj["log_probs"].append(total_lp)
            traj["values"].append(float(value_t.item()))
            traj["rewards"].append(float(reward))
            traj["dones"].append(done)

        final_bot_prob, final_anom = _score_sequence_with_warden(
            warden, delays_out, channels_out, self.device
        )
        ep_stats = {
            "reward": float(sum(traj["rewards"])),
            "final_bot_prob": final_bot_prob,
            "final_anom": final_anom,
            "mean_delay": float(np.mean(delays_out)),
            "mean_kl": total_kl / max(num_items, 1),
            "burst_pen": total_burst_pen,
        }
        return traj, ep_stats

    def _compute_gae(
        self, rewards: Sequence[float], values: Sequence[float], dones: Sequence[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < n else 0.0)
            non_term = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * next_val * non_term - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_term * gae
            adv[t] = gae
        ret = adv + np.asarray(values, dtype=np.float32)
        return ret, adv

    def train(
        self,
        warden: nn.Module,
        num_updates: int = 40,
        episodes_per_update: int = 8,
        carriers_per_episode: int = 50,
        hours_pool: Optional[Sequence[float]] = None,
        seed: int = 42,
    ) -> List[dict]:
        """Train with multi-episode minibatch PPO updates."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        warden.to(self.device)
        warden.eval()

        history: List[dict] = []
        best_score = -float("inf")
        best_state: Optional[dict] = None
        for upd in range(num_updates):
            batch_lists: dict[str, list] = {
                k: []
                for k in (
                    "states",
                    "prior_mix_logits",
                    "prior_means",
                    "prior_scales",
                    "prior_ch_logits",
                    "ch_masks",
                    "comp_actions",
                    "log_delay_actions",
                    "ch_actions",
                    "log_probs",
                    "values",
                    "returns",
                    "advantages",
                )
            }
            ep_metrics = []
            for _ in range(episodes_per_update):
                if hours_pool:
                    hour = float(hours_pool[int(rng.integers(0, len(hours_pool)))])
                else:
                    hour = float(rng.integers(0, 24))
                traj, stats = self._collect_episode(
                    warden,
                    num_items=carriers_per_episode,
                    start_hour=hour,
                )
                ret, adv = self._compute_gae(
                    traj["rewards"], traj["values"], traj["dones"]
                )
                for key in (
                    "states",
                    "prior_mix_logits",
                    "prior_means",
                    "prior_scales",
                    "prior_ch_logits",
                    "ch_masks",
                    "comp_actions",
                    "log_delay_actions",
                    "ch_actions",
                    "log_probs",
                    "values",
                ):
                    batch_lists[key].extend(traj[key])
                batch_lists["returns"].extend(ret.tolist())
                batch_lists["advantages"].extend(adv.tolist())
                ep_metrics.append(stats)

            # Convert rollout batch to tensors
            states_t = torch.tensor(np.asarray(batch_lists["states"]), dtype=torch.float32, device=self.device)
            pmix_t = torch.tensor(np.asarray(batch_lists["prior_mix_logits"]), dtype=torch.float32, device=self.device)
            pmean_t = torch.tensor(np.asarray(batch_lists["prior_means"]), dtype=torch.float32, device=self.device)
            pscale_t = torch.tensor(np.asarray(batch_lists["prior_scales"]), dtype=torch.float32, device=self.device)
            pch_t = torch.tensor(np.asarray(batch_lists["prior_ch_logits"]), dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(np.asarray(batch_lists["ch_masks"]), dtype=torch.float32, device=self.device)
            comp_t = torch.tensor(batch_lists["comp_actions"], dtype=torch.long, device=self.device)
            logd_t = torch.tensor(batch_lists["log_delay_actions"], dtype=torch.float32, device=self.device)
            ch_t = torch.tensor(batch_lists["ch_actions"], dtype=torch.long, device=self.device)
            old_lp_t = torch.tensor(batch_lists["log_probs"], dtype=torch.float32, device=self.device)
            old_val_t = torch.tensor(batch_lists["values"], dtype=torch.float32, device=self.device)
            ret_t = torch.tensor(batch_lists["returns"], dtype=torch.float32, device=self.device)
            adv_t = torch.tensor(batch_lists["advantages"], dtype=torch.float32, device=self.device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            self.actor_critic.train()
            n_total = states_t.size(0)
            mb = min(self.cfg.minibatch_size, n_total)
            pol_losses, val_losses, kls = [], [], []

            for _ in range(self.cfg.ppo_epochs):
                perm = torch.randperm(n_total, device=self.device)
                for start in range(0, n_total, mb):
                    idx = perm[start : start + mb]
                    new_lp, vals, ent, kl = self.actor_critic.evaluate_actions(
                        states_t[idx],
                        pmix_t[idx],
                        pmean_t[idx],
                        pscale_t[idx],
                        pch_t[idx],
                        mask_t[idx],
                        comp_t[idx],
                        logd_t[idx],
                        ch_t[idx],
                    )
                    ratio = torch.exp((new_lp - old_lp_t[idx]).clamp(-10.0, 10.0))
                    s1 = ratio * adv_t[idx]
                    s2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_t[idx]
                    pol_loss = -torch.min(s1, s2).mean()

                    # Value clipping
                    v_clipped = old_val_t[idx] + torch.clamp(
                        vals - old_val_t[idx], -self.cfg.clip_eps, self.cfg.clip_eps
                    )
                    v_loss = 0.5 * torch.max(
                        (vals - ret_t[idx]).pow(2), (v_clipped - ret_t[idx]).pow(2)
                    ).mean()

                    loss = (
                        pol_loss
                        + self.cfg.value_coef * v_loss
                        - self.cfg.entropy_coef * ent.mean()
                        + self.cfg.kl_loss_coef * kl.mean()
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.cfg.max_grad_norm
                    )
                    self.optimizer.step()

                    pol_losses.append(float(pol_loss.item()))
                    val_losses.append(float(v_loss.item()))
                    kls.append(float(kl.mean().item()))

            row = {
                "update": upd + 1,
                "mean_reward": float(np.mean([m["reward"] for m in ep_metrics])),
                "mean_bot_prob": float(np.mean([m["final_bot_prob"] for m in ep_metrics])),
                "mean_anom": float(np.mean([m["final_anom"] for m in ep_metrics])),
                "mean_delay": float(np.mean([m["mean_delay"] for m in ep_metrics])),
                "mean_kl": float(np.mean([m["mean_kl"] for m in ep_metrics])),
                "policy_loss": float(np.mean(pol_losses)),
                "value_loss": float(np.mean(val_losses)),
            }
            history.append(row)
            # Track best reward checkpoint while keeping KL anchored near the TPP prior
            if row["mean_reward"] > best_score and row["mean_kl"] <= 0.25:
                best_score = row["mean_reward"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.actor_critic.state_dict().items()
                }
            if (upd + 1) % max(1, num_updates // 5) == 0 or upd == 0 or (upd + 1) == num_updates:
                print(
                    f"  [PPO upd {upd + 1:02d}/{num_updates}] "
                    f"R={row['mean_reward']:.2f} | bot={row['mean_bot_prob']:.3f} | "
                    f"anom={row['mean_anom']:.3f} | delay={row['mean_delay']:.2f}s | "
                    f"KL={row['mean_kl']:.3f}"
                )
        if best_state is not None:
            self.actor_critic.load_state_dict(best_state)
            self.actor_critic.to(self.device)
        return history

    def save_checkpoint(self, path: Path | str, extra: Optional[dict] = None) -> None:
        payload = {
            "model_type": "prior_guided_tpp_ppo",
            "ppo_config": self.cfg.to_dict(),
            "tpp_config": self.tpp.config.to_dict(),
            "tpp_state_dict": self.tpp.state_dict(),
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "cooldowns": {str(k): float(v) for k, v in self.cooldowns.items()},
        }
        if extra:
            payload.update(extra)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(out))

    @classmethod
    def load_checkpoint(
        cls, path: Path | str, device: str = "cpu"
    ) -> "PriorGuidedPPOAgent":
        ckpt = load_trained_checkpoint(str(path), map_location=device)
        from src.stealth.tpp.model import TPPConfig

        tpp_cfg = TPPConfig.from_dict(ckpt["tpp_config"])
        tpp = TemporalPointProcessGenerator(tpp_cfg)
        tpp.load_state_dict(ckpt["tpp_state_dict"])

        ppo_cfg = PriorGuidedPPOConfig.from_dict(ckpt["ppo_config"])
        ppo_cfg.device = device
        cooldowns = {int(k): float(v) for k, v in ckpt.get("cooldowns", {"0": 6.0, "1": 12.0, "2": 4.0}).items()}

        agent = cls(tpp=tpp, config=ppo_cfg, cooldowns=cooldowns)
        agent.actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
        agent.actor_critic.to(device)
        agent.actor_critic.eval()
        return agent
