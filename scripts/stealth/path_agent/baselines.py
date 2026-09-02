import numpy as np

# Feature indices, matching network_env.FEATURES.
_LATENCY_IDX = 0


class RandomPolicy:
    def __init__(self, num_paths: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.num_paths = num_paths

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, obs.shape[0]))


class GreedyLatencyPolicy:
    """Always picks whichever path currently reports the lowest latency."""

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmin(obs[:, _LATENCY_IDX]))


class RoundRobinPolicy:
    def __init__(self, num_paths: int):
        self.num_paths = num_paths
        self._next = 0

    def act(self, obs: np.ndarray) -> int:
        action = self._next % obs.shape[0]
        self._next += 1
        return action
