import numpy as np

from path_agent.config import Config

FEATURES = ["latency_norm", "queue_occupancy", "loss_rate", "utilization", "capacity_norm"]
NUM_FEATURES = len(FEATURES)

_LATENCY_NORM_MS = 200.0   # latency at/above this is treated as "fully congested" for the observation
_UTIL_NORM = 2.0           # utilization at/above this is treated as "fully congested" for the observation


class NetworkEnv:
    """Simulates `num_paths` parallel links to one destination.

    Each step the agent sends its flow's demand down exactly one chosen path.
    Every path also carries independent background cross-traffic that evolves
    over time (mean-reverting + occasional bursts), so congestion is temporally
    correlated and the agent's own routing choices feed back into what it
    observes next (queues persist across steps).
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.n = config.num_paths
        self.rng = np.random.default_rng(config.seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.cfg
        n = self.n
        lo, hi = cfg.capacity_range
        self.capacity = self.rng.uniform(lo, hi, size=n)
        lo, hi = cfg.base_latency_range
        self.base_latency = self.rng.uniform(lo, hi, size=n)
        lo, hi = cfg.buffer_range
        self.buffer = self.rng.uniform(lo, hi, size=n)

        self.queue = np.zeros(n)
        self.background_frac = np.full(n, cfg.background_mean_frac)
        self.burst_timer = np.zeros(n, dtype=int)

        self.t = 0
        self._last_loss_rate = np.zeros(n)
        return self._observe()

    def _update_background(self):
        cfg = self.cfg
        n = self.n

        # Ornstein-Uhlenbeck mean reversion toward the target background load fraction.
        noise = self.rng.normal(0.0, cfg.background_volatility, size=n)
        self.background_frac += cfg.background_reversion * (cfg.background_mean_frac - self.background_frac) + noise
        self.background_frac = np.clip(self.background_frac, 0.0, 1.5)

        # Start new bursts.
        new_bursts = (self.burst_timer <= 0) & (self.rng.random(n) < cfg.burst_prob)
        self.burst_timer[new_bursts] = cfg.burst_duration

        effective_frac = self.background_frac.copy()
        bursting = self.burst_timer > 0
        effective_frac[bursting] += cfg.burst_magnitude_frac
        self.burst_timer[bursting] -= 1

        return effective_frac

    def step(self, action: int):
        assert 0 <= action < self.n
        cfg = self.cfg

        effective_bg_frac = self._update_background()
        background_arrival = effective_bg_frac * self.capacity

        agent_arrival = np.zeros(self.n)
        agent_arrival[action] = cfg.agent_demand

        total_arrival = background_arrival + agent_arrival

        unclamped_queue = np.maximum(0.0, self.queue + total_arrival * cfg.dt - self.capacity * cfg.dt)
        overflow = np.maximum(0.0, unclamped_queue - self.buffer)
        self.queue = np.minimum(unclamped_queue, self.buffer)

        offered = total_arrival * cfg.dt
        loss_rate = np.divide(overflow, offered, out=np.zeros_like(offered), where=offered > 1e-8)
        self._last_loss_rate = loss_rate

        utilization = total_arrival / self.capacity
        queue_delay_ms = (self.queue / self.capacity) * 1000.0
        latency = self.base_latency + queue_delay_ms

        reward = -(latency[action] / cfg.latency_scale) - cfg.loss_penalty * loss_rate[action]

        self.t += 1
        done = self.t >= cfg.episode_length

        info = {
            "latency": latency.copy(),
            "loss_rate": loss_rate.copy(),
            "utilization": utilization.copy(),
            "chosen_latency": latency[action],
            "chosen_loss_rate": loss_rate[action],
        }
        obs = self._observe(latency=latency, loss_rate=loss_rate, utilization=utilization)
        return obs, reward, done, info

    def raw_path_stats(self):
        """Snapshot of current per-path congestion in the raw (unnormalized)
        format expected by observation.build_observation - i.e. what a real
        congestion source would report. Reflects state *before* any new
        agent packet is added, matching what should be measured prior to a
        routing decision.
        """
        latency = self.base_latency + (self.queue / self.capacity) * 1000.0
        arrival_rate = self.background_frac * self.capacity
        return [
            {
                "latency_ms": float(latency[i]),
                "queue": float(self.queue[i]),
                "buffer": float(self.buffer[i]),
                "loss_rate": float(self._last_loss_rate[i]),
                "arrival_rate": float(arrival_rate[i]),
                "capacity": float(self.capacity[i]),
            }
            for i in range(self.n)
        ]

    def _observe(self, latency=None, loss_rate=None, utilization=None):
        cfg = self.cfg
        if latency is None:
            latency = self.base_latency + (self.queue / self.capacity) * 1000.0
        if loss_rate is None:
            loss_rate = self._last_loss_rate
        if utilization is None:
            utilization = self.background_frac

        latency_norm = np.clip(latency / _LATENCY_NORM_MS, 0.0, 1.0)
        queue_occupancy = np.clip(self.queue / self.buffer, 0.0, 1.0)
        loss_rate_c = np.clip(loss_rate, 0.0, 1.0)
        utilization_norm = np.clip(utilization / _UTIL_NORM, 0.0, 1.0)
        capacity_max = cfg.capacity_range[1]
        capacity_norm = np.clip(self.capacity / capacity_max, 0.0, 1.0)

        obs = np.stack(
            [latency_norm, queue_occupancy, loss_rate_c, utilization_norm, capacity_norm],
            axis=-1,
        ).astype(np.float32)
        return obs


def _self_check():
    cfg = Config(num_paths=4, episode_length=500, seed=42)
    env = NetworkEnv(cfg)
    obs = env.reset()
    assert obs.shape == (4, NUM_FEATURES)

    saw_loss = False
    for _ in range(cfg.episode_length):
        obs, reward, done, info = env.step(0)  # hammer path 0 to force saturation
        assert np.all(env.queue >= 0.0) and np.all(env.queue <= env.buffer + 1e-6)
        assert np.all(info["latency"] >= env.base_latency - 1e-6)
        assert not np.any(np.isnan(obs))
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        if info["loss_rate"][0] > 0:
            saw_loss = True
        if done:
            env.reset()

    assert saw_loss, "expected path 0 to saturate and drop packets under sustained load"
    print("network_env self-check passed")


if __name__ == "__main__":
    _self_check()
