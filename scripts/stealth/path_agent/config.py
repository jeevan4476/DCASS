from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

_MODULE_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # --- environment topology ---
    num_paths: int = 4
    seed: int = 0

    # per-path physical parameters are sampled uniformly at reset() within these ranges
    capacity_range: Tuple[float, float] = (50.0, 150.0)     # units/sec
    base_latency_range: Tuple[float, float] = (5.0, 40.0)   # ms
    buffer_range: Tuple[float, float] = (100.0, 400.0)      # queue capacity, units

    # --- traffic dynamics ---
    dt: float = 1.0                       # simulated seconds per step
    agent_demand: float = 40.0            # units/sec injected onto the chosen path
    background_mean_frac: float = 0.4     # background load as fraction of capacity, mean-reversion target
    background_volatility: float = 0.08   # OU process noise scale (fraction of capacity per step)
    background_reversion: float = 0.15    # OU mean-reversion rate
    burst_prob: float = 0.03              # probability per path per step of a burst event
    burst_magnitude_frac: float = 0.6     # extra background load (fraction of capacity) during a burst
    burst_duration: int = 5               # steps a burst lasts

    episode_length: int = 200

    # --- reward shaping ---
    latency_scale: float = 100.0          # normalizes latency into a roughly [0,1]-ish penalty
    loss_penalty: float = 2.0

    # --- DQN hyperparameters ---
    hidden_dim: int = 64
    lr: float = 1e-3
    gamma: float = 0.95
    buffer_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 1_000
    target_sync_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000

    # --- training loop ---
    num_episodes: int = 500
    log_every: int = 10
    checkpoint_path: str = str(_MODULE_DIR / "runs" / "dqn.pt")
    log_csv_path: str = str(_MODULE_DIR / "runs" / "train_log.csv")
