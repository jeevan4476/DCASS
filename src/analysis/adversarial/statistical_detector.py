"""Independent non-neural traffic detector (Gradient Boosted Trees).

Provides a second, architecturally unrelated detector family alongside the
BiLSTM-Transformer DPI Warden so traffic evasion results cannot be dismissed
as overfitting to one neural architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def _log_bin_entropy(delays: np.ndarray, bins: int = 50) -> float:
    clipped = np.clip(delays, 0.01, 10000.0)
    edges = np.logspace(np.log10(0.01), np.log10(10000.0), bins + 1)
    hist, _ = np.histogram(clipped, bins=edges)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _autocorr(arr: np.ndarray, lag: int) -> float:
    if len(arr) <= lag + 1:
        return 0.0
    x = arr[:-lag]
    y = arr[lag:]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    return float(np.clip(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy), -1.0, 1.0))


def extract_traffic_feature_vector(
    delays: Sequence[float],
    channels: Optional[Sequence[int]] = None,
    *,
    num_channels: int = 3,
    ignore_channels: bool = False,
) -> np.ndarray:
    """Extract a 24-dim statistical timing (and optional channel) fingerprint."""
    d = np.asarray(delays, dtype=np.float64)
    if d.size == 0:
        d = np.array([1.0], dtype=np.float64)
    d = np.clip(d, 0.05, 10000.0)
    ld = np.log1p(d)

    mu = float(d.mean())
    sd = float(d.std())
    cv = sd / (mu + 1e-8)

    qs = np.quantile(d, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    iqr = qs[4] - qs[2]
    mad = float(np.median(np.abs(d - qs[3])))

    centered = d - mu
    skew = float(np.mean(centered**3) / (sd**3 + 1e-8))
    kurt = float(np.mean(centered**4) / (sd**4 + 1e-8))

    # Sliding 5s burst statistics
    ts = np.cumsum(d)
    burst_counts = np.array([np.sum((ts >= t) & (ts < t + 5.0)) for t in ts], dtype=np.float64)
    max_burst = float(burst_counts.max()) if burst_counts.size else 0.0
    frac_burst4 = float(np.mean(burst_counts >= 4)) if burst_counts.size else 0.0

    # Lag-1, lag-2 autocorrelation in log-delay space
    ac1 = _autocorr(ld, 1)
    ac2 = _autocorr(ld, 2)
    diff_std = float(np.std(np.diff(ld))) if len(ld) > 1 else 0.0

    timing_feats = [
        np.log1p(mu),
        np.log1p(sd),
        min(cv, 10.0),
        np.arcsinh(np.clip(skew, -1e4, 1e4)),
        np.arcsinh(np.clip(kurt, -1e4, 1e4)),
        _log_bin_entropy(d),
        np.log1p(qs[0]),
        np.log1p(qs[1]),
        np.log1p(qs[2]),
        np.log1p(qs[3]),
        np.log1p(qs[4]),
        np.log1p(qs[5]),
        np.log1p(qs[6]),
        np.log1p(max(iqr, 0.0)),
        np.log1p(max(mad, 0.0)),
        max_burst,
        frac_burst4,
        float(np.mean(d <= 1.0)),
        float(np.mean(d > 30.0)),
        ac1,
        ac2,
        diff_std,
    ]

    if ignore_channels or channels is None or len(channels) == 0:
        return np.asarray(timing_feats, dtype=np.float64)

    ch = np.asarray(channels, dtype=np.int64) % num_channels
    counts = np.bincount(ch, minlength=num_channels).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    ch_ent = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
    switch_rate = float(np.mean(ch[1:] != ch[:-1])) if len(ch) > 1 else 0.0
    return np.asarray(timing_feats + [ch_ent, switch_rate] + probs.tolist(), dtype=np.float64)


@dataclass
class StatisticalTrafficDetector:
    """Gradient Boosted Trees traffic detector (architecturally distinct from Warden)."""

    ignore_channels: bool = False
    num_channels: int = 3
    seed: int = 42
    max_iter: int = 200
    max_depth: int = 6
    learning_rate: float = 0.06
    model: Optional[HistGradientBoostingClassifier] = None

    def _featurize(self, sessions: Sequence[dict]) -> np.ndarray:
        return np.vstack(
            [
                extract_traffic_feature_vector(
                    s["delays"][:64],
                    s.get("channels", [])[:64],
                    num_channels=self.num_channels,
                    ignore_channels=self.ignore_channels,
                )
                for s in sessions
            ]
        )

    def fit(self, sessions: Sequence[dict], labels: Sequence[int]) -> "StatisticalTrafficDetector":
        x = self._featurize(sessions)
        y = np.asarray(labels, dtype=np.int64)
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=1.0,
            random_state=self.seed,
        )
        self.model.fit(x, y)
        return self

    def predict_bot_proba(self, sessions: Sequence[dict]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("StatisticalTrafficDetector must be fitted before scoring")
        x = self._featurize(sessions)
        return self.model.predict_proba(x)[:, 1].astype(np.float64)

    def score_sequence(
        self, delays: Sequence[float], channels: Optional[Sequence[int]] = None
    ) -> float:
        if self.model is None:
            raise RuntimeError("StatisticalTrafficDetector must be fitted before scoring")
        x = extract_traffic_feature_vector(
            delays[:64],
            None if channels is None else channels[:64],
            num_channels=self.num_channels,
            ignore_channels=self.ignore_channels,
        ).reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])
