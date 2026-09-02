"""A WGAN-GP that learns the rhythm of an event log and generates N random timestamps.

Usage
-----
    python gan.py train  [--steps 3000] [--samples 20000]
    python gan.py sample -n 20 [--start 2026-09-01] [--end 2026-10-01] [--csv out.csv]
    python gan.py eval

The generator maps a noise vector to a point on the unit circle, which decodes to a
position within the week (weekday + time of day). The calendar week itself is drawn
uniformly from the requested date range, so every call yields fresh timestamps.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LATENT = 16
HIDDEN = 256
HARMONICS = 8  # critic Fourier features per circle, to counter MLP spectral bias
WEEK_SECONDS = 7 * 24 * 3600
DAY_SECONDS = 24 * 3600
DEFAULT_CKPT = "gan_ckpt.pt"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- real data


def weekly_intensity(u: np.ndarray) -> np.ndarray:
    """Relative event rate at position u in [0, 1) within the week."""
    weekday = np.floor(u * 7).astype(int)          # 0 = Monday
    hour = (u * 7 % 1.0) * 24.0
    weekend = weekday >= 5

    business = (
        1.0 * np.exp(-0.5 * ((hour - 10.0) / 1.6) ** 2)     # morning peak
        + 0.85 * np.exp(-0.5 * ((hour - 15.0) / 1.8) ** 2)  # afternoon peak
        - 0.45 * np.exp(-0.5 * ((hour - 13.0) / 0.7) ** 2)  # lunch dip
    )
    leisure = 0.55 * np.exp(-0.5 * ((hour - 17.0) / 3.0) ** 2)

    rate = np.where(weekend, leisure, business)
    return np.clip(rate, 0.0, None) + 0.02  # noise floor: no hour is impossible


def make_real_timestamps(n: int = 20000, weeks: int = 8, seed: int = 0) -> pd.DatetimeIndex:
    """Rejection-sample n event times from `weekly_intensity` over `weeks` weeks."""
    rng = np.random.default_rng(seed)
    peak = weekly_intensity(np.linspace(0.0, 1.0, 100_001)).max()

    kept: list[np.ndarray] = []
    total = 0
    while total < n:
        batch = max(n * 4, 4096)
        u = rng.random(batch)
        accept = rng.random(batch) * peak < weekly_intensity(u)
        u = u[accept]
        kept.append(u)
        total += u.size
    u = np.concatenate(kept)[:n]

    start = pd.Timestamp("2026-01-05")  # a Monday
    week = rng.integers(0, weeks, size=n)
    offsets = (week * WEEK_SECONDS + u * WEEK_SECONDS).astype(np.int64)
    return pd.DatetimeIndex(start + pd.to_timedelta(offsets, unit="s")).sort_values()


# ---------------------------------------------------------------- encoding


def week_position(index: pd.DatetimeIndex) -> np.ndarray:
    """Position within the week in [0, 1): Monday 00:00 -> 0."""
    seconds = (
        index.dayofweek.to_numpy() * 86400
        + index.hour.to_numpy() * 3600
        + index.minute.to_numpy() * 60
        + index.second.to_numpy()
    )
    return seconds / WEEK_SECONDS


def to_features(index: pd.DatetimeIndex) -> torch.Tensor:
    """Timestamps -> [N, 4]: a point on the week-circle and one on the day-circle.

    Two circles, not one. On a single week-circle an hour spans only 2pi/168 radians,
    so the gradient penalty (which normalises gradients at scale 1) flattens all
    time-of-day detail away and only the weekday rhythm survives. Giving the day its
    own full circle puts both rhythms at the same scale.
    """
    u = week_position(index)
    theta_w = 2.0 * math.pi * u
    theta_d = 2.0 * math.pi * ((u * 7.0) % 1.0)
    stacked = np.stack(
        [np.cos(theta_w), np.sin(theta_w), np.cos(theta_d), np.sin(theta_d)], axis=1
    )
    return torch.tensor(stacked, dtype=torch.float32)


def decode_weekday_time(feats: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """[N, 4] -> (weekday in 0..6, time-of-day fraction in [0, 1))."""
    xy = feats.detach().cpu().numpy()
    u_coarse = np.mod(np.arctan2(xy[:, 1], xy[:, 0]) / (2.0 * math.pi), 1.0)
    time_frac = np.mod(np.arctan2(xy[:, 3], xy[:, 2]) / (2.0 * math.pi), 1.0)
    weekday = np.clip(np.rint(u_coarse * 7.0 - time_frac), 0.0, 6.0).astype(np.int64)
    return weekday, time_frac


def candidate_timestamps(
    feats: torch.Tensor,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """[N, 4] circle points -> (candidate timestamps, mask of which land in [start, end)).

    Every whole day touched by [start, end) is a candidate date. Each sample's weekday
    (from the generator) picks among the candidate dates that share it; if the range is
    too short to contain that weekday at all, it falls back to any candidate date. The
    time-of-day is taken as-is from the generator and NOT clamped into range — a window
    narrower than a day (e.g. one hour) would otherwise bunch every rejected point onto
    the boundary. Instead the caller re-draws rejected samples (see generate_timestamps).
    """
    weekday, time_frac = decode_weekday_time(feats)

    days = pd.date_range(start.normalize(), end.ceil("D"), freq="D", inclusive="left")
    if len(days) == 0:
        days = pd.DatetimeIndex([start.normalize()])
    day_weekday = days.dayofweek.to_numpy()

    day_values = days.values.astype("datetime64[s]")
    buckets = [np.flatnonzero(day_weekday == d) for d in range(7)]
    day_start = np.empty(len(weekday), dtype="datetime64[s]")
    for d in range(7):
        mask = weekday == d
        if not mask.any():
            continue
        pool = buckets[d] if buckets[d].size else np.arange(len(days))
        day_start[mask] = day_values[pool[rng.integers(0, pool.size, size=mask.sum())]]

    offsets = (time_frac * DAY_SECONDS).astype(np.int64)
    cand = day_start + offsets.astype("timedelta64[s]")

    lo, hi = np.datetime64(start, "s"), np.datetime64(end, "s")
    valid = (cand >= lo) & (cand < hi)
    return cand, valid


# ---------------------------------------------------------------- models


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, 4),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Project each half onto its unit circle: the real data lives there exactly.
        raw = self.net(z)
        week = nn.functional.normalize(raw[:, :2], dim=1, eps=1e-8)
        day = nn.functional.normalize(raw[:, 2:], dim=1, eps=1e-8)
        return torch.cat([week, day], dim=1)


def harmonics(x: torch.Tensor, k_max: int = HARMONICS) -> torch.Tensor:
    """[N, 4] -> [N, 4 + 4*k_max]: each circle expanded into (cos kθ, sin kθ).

    Computed as powers of z = x0 + i·x1, so on the unit circle z^k is exactly
    (cos kθ, sin kθ). A plain MLP is biased toward low frequencies and would miss the
    sharper features of the daily shape; these hand it those frequencies directly.
    """
    z = torch.view_as_complex(x.contiguous().view(-1, 2, 2))  # [N, 2] complex
    powers = torch.stack([z**k for k in range(1, k_max + 1)], dim=-1)
    return torch.cat([x, torch.view_as_real(powers).flatten(1)], dim=1)


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 4 * HARMONICS, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(harmonics(x))


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    eps = torch.rand(real.size(0), 1, device=real.device)
    mixed = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    scores = critic(mixed)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=mixed,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
    )[0]
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


# ---------------------------------------------------------------- training


def train(
    steps: int = 3000,
    samples: int = 20000,
    batch: int = 256,
    n_critic: int = 5,
    lr: float = 1e-4,
    gp_weight: float = 10.0,
    seed: int = 0,
    ckpt: str = DEFAULT_CKPT,
) -> str:
    torch.manual_seed(seed)
    dev = device()

    real_index = make_real_timestamps(samples, seed=seed)
    real = to_features(real_index).to(dev)

    gen, critic = Generator().to(dev), Critic().to(dev)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

    print(f"training on {dev} | {samples} real timestamps | {steps} generator steps")
    for step in range(1, steps + 1):
        for _ in range(n_critic):
            idx = torch.randint(0, real.size(0), (batch,), device=dev)
            real_batch = real[idx]
            with torch.no_grad():
                fake = gen(torch.randn(batch, LATENT, device=dev))
            w_est = critic(real_batch).mean() - critic(fake).mean()
            loss_d = -w_est + gp_weight * gradient_penalty(critic, real_batch, fake)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

        loss_g = -critic(gen(torch.randn(batch, LATENT, device=dev))).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step % 250 == 0 or step == 1:
            print(f"  step {step:5d}  wasserstein~{w_est.item():+.4f}  loss_g {loss_g.item():+.4f}")

    torch.save({"generator": gen.state_dict(), "latent": LATENT, "hidden": HIDDEN}, ckpt)
    print(f"saved {ckpt}")
    return ckpt


def load_generator(ckpt: str = DEFAULT_CKPT, auto_train: bool = True) -> Generator:
    if not os.path.exists(ckpt):
        if not auto_train:
            raise FileNotFoundError(f"no checkpoint at {ckpt} — run `python gan.py train` first")
        print(f"no checkpoint at {ckpt}; training one now...")
        train(ckpt=ckpt)
    state = torch.load(ckpt, map_location=device())
    gen = Generator().to(device())
    gen.load_state_dict(state["generator"])
    gen.eval()
    return gen


# ---------------------------------------------------------------- sampling


def generate_timestamps(
    n: int,
    start: str | pd.Timestamp = "2026-09-01",
    end: str | pd.Timestamp = "2026-10-01",
    ckpt: str = DEFAULT_CKPT,
    seed: int | None = None,
) -> pd.DatetimeIndex:
    """Draw n randomised timestamps in [start, end) from the trained generator.

    Uses rejection sampling: draws are decoded to a candidate timestamp and kept only
    if it lands inside [start, end); rejects are replaced by fresh draws. This lets the
    window be as narrow as a single hour (or less) while still following the learned
    time-of-day shape within it, rather than piling rejected points onto the edges.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    gen = load_generator(ckpt)
    rng = np.random.default_rng(seed)

    collected: list[np.ndarray] = []
    remaining = n
    batch = max(n * 4, 512)
    for _ in range(200):  # generous cap; the noise floor guarantees nonzero acceptance
        if remaining <= 0:
            break
        with torch.no_grad():
            feats = gen(torch.randn(batch, LATENT, device=device()))
        cand, valid = candidate_timestamps(feats, start_ts, end_ts, rng)
        picked = cand[valid][:remaining]
        collected.append(picked)
        remaining -= len(picked)
        batch = min(batch * 2, 50_000)  # ramp up if the window is rare
    else:
        raise RuntimeError(
            "could not draw enough timestamps in this window — it may be too narrow "
            "or fall in an almost-never-occurring part of the learned pattern"
        )

    return pd.DatetimeIndex(np.concatenate(collected)).sort_values()


# ---------------------------------------------------------------- evaluation


def histogram(counts: np.ndarray, labels: list[str], width: int = 40) -> str:
    share = counts / max(counts.sum(), 1)
    peak = max(share.max(), 1e-9)
    lines = [
        f"  {label:>3}  {'#' * int(round(width * s / peak)):<{width}} {s * 100:5.1f}%"
        for label, s in zip(labels, share)
    ]
    return "\n".join(lines)


def evaluate(ckpt: str = DEFAULT_CKPT, n: int = 20000) -> None:
    real = make_real_timestamps(n)
    fake = generate_timestamps(n, ckpt=ckpt)

    hours = [f"{h:02d}" for h in range(24)]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for name, title, labels, values in (
        ("hour", "hour of day", hours, (real.hour.to_numpy(), fake.hour.to_numpy())),
        ("day", "weekday", days, (real.dayofweek.to_numpy(), fake.dayofweek.to_numpy())),
    ):
        bins = len(labels)
        r = np.bincount(values[0], minlength=bins)[:bins]
        f = np.bincount(values[1], minlength=bins)[:bins]
        print(f"\nREAL - {title}\n{histogram(r, labels)}")
        print(f"\nGENERATED - {title}\n{histogram(f, labels)}")
        tvd = 0.5 * np.abs(r / r.sum() - f / f.sum()).sum()
        print(f"\n  total variation distance ({name}): {tvd:.4f}  (0 = identical)")


# ---------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the GAN on a synthetic event log")
    p_train.add_argument("--steps", type=int, default=3000)
    p_train.add_argument("--samples", type=int, default=20000)
    p_train.add_argument("--batch", type=int, default=256)
    p_train.add_argument("--n-critic", type=int, default=5)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--ckpt", default=DEFAULT_CKPT)

    p_sample = sub.add_parser("sample", help="generate n randomised timestamps")
    p_sample.add_argument("-n", type=int, required=True, help="how many timestamps")
    p_sample.add_argument("--start", default="2026-09-01")
    p_sample.add_argument("--end", default="2026-10-01")
    p_sample.add_argument("--csv", help="also write the timestamps to this CSV")
    p_sample.add_argument("--seed", type=int, default=None)
    p_sample.add_argument("--ckpt", default=DEFAULT_CKPT)

    p_eval = sub.add_parser("eval", help="compare real vs generated distributions")
    p_eval.add_argument("-n", type=int, default=20000)
    p_eval.add_argument("--ckpt", default=DEFAULT_CKPT)

    args = parser.parse_args()

    if args.command == "train":
        train(
            steps=args.steps,
            samples=args.samples,
            batch=args.batch,
            n_critic=args.n_critic,
            lr=args.lr,
            seed=args.seed,
            ckpt=args.ckpt,
        )
    elif args.command == "sample":
        ts = generate_timestamps(args.n, args.start, args.end, ckpt=args.ckpt, seed=args.seed)
        for t in ts:
            print(t.strftime("%Y-%m-%d %H:%M:%S  (%a)"))
        if args.csv:
            pd.DataFrame({"timestamp": ts}).to_csv(args.csv, index=False)
            print(f"\nwrote {len(ts)} timestamps to {args.csv}")
    else:
        evaluate(ckpt=args.ckpt, n=args.n)


if __name__ == "__main__":
    main()
