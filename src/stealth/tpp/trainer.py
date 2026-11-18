"""Training loop for the Conditional Autoregressive Log-Normal Mixture TPP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model import TPPConfig, TemporalPointProcessGenerator


@dataclass
class TPPTrainingConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_seq_len: int = 64
    channel_weight: float = 0.35
    seed: int = 42
    balance_contributors: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TPPSessionDataset(Dataset):
    """Dataset of human sessions padded to max_seq_len with validity masks."""

    def __init__(
        self,
        sessions: Sequence[dict],
        max_seq_len: int = 64,
        min_delay: float = 0.25,
        max_delay: float = 120.0,
    ):
        self.sessions = list(sessions)
        self.max_seq_len = int(max_seq_len)
        self.min_delay = float(min_delay)
        self.max_delay = float(max_delay)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.sessions[idx]
        raw_delays = np.asarray(s["delays"][: self.max_seq_len], dtype=np.float32)
        raw_channels = np.asarray(s["channels"][: self.max_seq_len], dtype=np.int64)
        n = len(raw_delays)

        delays = np.full(self.max_seq_len, self.min_delay, dtype=np.float32)
        channels = np.zeros(self.max_seq_len, dtype=np.int64)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)

        if n > 0:
            delays[:n] = np.clip(raw_delays, self.min_delay, self.max_delay)
            channels[:n] = raw_channels
            mask[:n] = 1.0

        tod = float(s.get("time_of_day", 12.0))
        return {
            "delays": torch.from_numpy(delays),
            "channels": torch.from_numpy(channels),
            "time_of_day": torch.tensor(tod, dtype=torch.float32),
            "mask": torch.from_numpy(mask),
        }


def _contributor_weights(sessions: Sequence[dict]) -> Optional[torch.DoubleTensor]:
    if not sessions or any(s.get("contributor_id") is None for s in sessions):
        return None
    counts: Dict[str, int] = {}
    for s in sessions:
        cid = str(s["contributor_id"])
        counts[cid] = counts.get(cid, 0) + 1
    weights = [1.0 / counts[str(s["contributor_id"])] for s in sessions]
    return torch.tensor(weights, dtype=torch.double)


def _empirical_log_stats(
    sessions: Sequence[dict], min_delay: float, max_delay: float
) -> tuple[float, float]:
    vals = []
    for s in sessions[:4000]:
        d = np.asarray(s["delays"], dtype=np.float64)
        if d.size:
            vals.append(np.log(np.clip(d, min_delay, max_delay)))
    if not vals:
        return 1.5, 1.2
    cat = np.concatenate(vals)
    return float(np.mean(cat)), max(float(np.std(cat)), 0.25)


class TPPTrainer:
    """Trains TemporalPointProcessGenerator with exact MLE and validation tracking."""

    def __init__(
        self,
        model: TemporalPointProcessGenerator,
        train_config: Optional[TPPTrainingConfig] = None,
    ):
        self.model = model
        self.cfg = train_config or TPPTrainingConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.history: List[dict] = []

    def _eval_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        tot_loss, tot_d, tot_c, n_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in loader:
                loss, d_nll, c_nll = self.model.compute_loss(
                    batch["delays"].to(self.device),
                    batch["channels"].to(self.device),
                    batch["time_of_day"].to(self.device),
                    batch["mask"].to(self.device),
                    channel_weight=self.cfg.channel_weight,
                )
                tot_loss += float(loss.item())
                tot_d += float(d_nll.item())
                tot_c += float(c_nll.item())
                n_batches += 1
        denom = max(n_batches, 1)
        return {
            "loss": tot_loss / denom,
            "delay_nll": tot_d / denom,
            "channel_nll": tot_c / denom,
        }

    def fit(
        self,
        train_sessions: Sequence[dict],
        val_sessions: Sequence[dict],
        cooldowns: Optional[Dict[int, float]] = None,
    ) -> dict:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        train_ds = TPPSessionDataset(
            train_sessions,
            max_seq_len=self.cfg.max_seq_len,
            min_delay=self.model.min_delay,
            max_delay=self.model.max_delay,
        )
        val_ds = TPPSessionDataset(
            val_sessions,
            max_seq_len=self.cfg.max_seq_len,
            min_delay=self.model.min_delay,
            max_delay=self.model.max_delay,
        )

        weights = _contributor_weights(train_sessions) if self.cfg.balance_contributors else None
        if weights is not None:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_ds),
                replacement=True,
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )
            train_loader = DataLoader(
                train_ds, batch_size=self.cfg.batch_size, sampler=sampler
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=self.cfg.batch_size, shuffle=True
            )
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.cfg.epochs, 1)
        )

        best_val_loss = float("inf")
        best_state: Optional[dict] = None
        best_epoch = -1

        for epoch in range(self.cfg.epochs):
            self.model.train()
            tr_loss, tr_d, tr_c, batches = 0.0, 0.0, 0.0, 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                loss, d_nll, c_nll = self.model.compute_loss(
                    batch["delays"].to(self.device),
                    batch["channels"].to(self.device),
                    batch["time_of_day"].to(self.device),
                    batch["mask"].to(self.device),
                    channel_weight=self.cfg.channel_weight,
                )
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                self.optimizer.step()
                tr_loss += float(loss.item())
                tr_d += float(d_nll.item())
                tr_c += float(c_nll.item())
                batches += 1

            scheduler.step()
            val_metrics = self._eval_loader(val_loader)
            row = {
                "epoch": epoch + 1,
                "train_loss": tr_loss / max(batches, 1),
                "train_delay_nll": tr_d / max(batches, 1),
                "train_channel_nll": tr_c / max(batches, 1),
                "val_loss": val_metrics["loss"],
                "val_delay_nll": val_metrics["delay_nll"],
                "val_channel_nll": val_metrics["channel_nll"],
            }
            self.history.append(row)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

            if (epoch + 1) % max(1, self.cfg.epochs // 5) == 0 or epoch == 0 or (epoch + 1) == self.cfg.epochs:
                print(
                    f"  [TPP epoch {epoch + 1:02d}/{self.cfg.epochs}] "
                    f"train={row['train_loss']:.4f} (d={row['train_delay_nll']:.4f}, c={row['train_channel_nll']:.4f}) | "
                    f"val={row['val_loss']:.4f} (d={row['val_delay_nll']:.4f}, c={row['val_channel_nll']:.4f})"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # Quick diagnostic KS against validation delays on routed schedules
        val_ref = np.concatenate(
            [
                np.clip(
                    np.asarray(s["delays"], dtype=np.float64),
                    self.model.min_delay,
                    self.model.max_delay,
                )
                for s in val_sessions[:500]
            ]
        )
        rng = np.random.default_rng(self.cfg.seed + 101)
        sampled_delays: list[float] = []
        val_hours = [float(s.get("time_of_day", 12.0)) for s in val_sessions[:200]]
        for idx in range(30):
            h = val_hours[idx % len(val_hours)] if val_hours else 12.0
            d_seq, _ = self.model.generate_routed_schedule(
                num_items=50,
                time_of_day=h,
                cooldowns=cooldowns,
                rng=rng,
                device=str(self.device),
            )
            sampled_delays.extend(d_seq)
        val_ks = float(ks_2samp(np.asarray(sampled_delays), val_ref).statistic)

        return {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "val_routed_ks": val_ks,
            "history": self.history,
        }


def train_tpp_model(
    train_sessions: Sequence[dict],
    val_sessions: Sequence[dict],
    model_config: Optional[TPPConfig] = None,
    train_config: Optional[TPPTrainingConfig] = None,
    cooldowns: Optional[Dict[int, float]] = None,
    out_path: Optional[Path | str] = None,
) -> tuple[TemporalPointProcessGenerator, dict]:
    """Fit a TemporalPointProcessGenerator on train_sessions and select via val_sessions."""
    cfg = model_config or TPPConfig()
    mean_ld, std_ld = _empirical_log_stats(train_sessions, cfg.min_delay, cfg.max_delay)
    cfg.log_delay_mean = mean_ld
    cfg.log_delay_std = std_ld

    model = TemporalPointProcessGenerator(cfg)
    trainer = TPPTrainer(model, train_config)
    summary = trainer.fit(train_sessions, val_sessions, cooldowns=cooldowns)
    if out_path is not None:
        model.save_checkpoint(out_path, extra={"training_summary": summary})
    return model, summary
