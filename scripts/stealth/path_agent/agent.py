import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from path_agent.config import Config
from path_agent.network_env import NUM_FEATURES


class PathScorer(nn.Module):
    """Scores each path independently with a shared MLP.

    Input:  [B, N, F] (batch, num_paths, features) - N can vary between calls
            since the same weights are applied per-path.
    Output: [B, N] Q-values, one per path.
    """

    def __init__(self, num_features: int = NUM_FEATURES, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, N, F] -> apply the same MLP to every row -> [B, N, 1] -> [B, N]
        return self.net(obs).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, num_paths: int, num_features: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, num_paths, num_features), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_paths, num_features), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
            self.done[idx],
        )

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, config: Config, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)
        self.online = PathScorer(NUM_FEATURES, config.hidden_dim).to(self.device)
        self.target = PathScorer(NUM_FEATURES, config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_capacity, config.num_paths, NUM_FEATURES)
        self.rng = np.random.default_rng(config.seed + 1)

        self.step_count = 0

    def epsilon(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self.step_count / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        num_paths = obs.shape[0]
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, num_paths))

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # [1, N, F]
            q = self.online(obs_t).squeeze(0)  # [N]
        return int(torch.argmax(q).item())

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self):
        cfg = self.cfg
        if len(self.buffer) < max(cfg.batch_size, cfg.warmup_steps):
            return None

        obs, action, reward, next_obs, done = self.buffer.sample(cfg.batch_size, self.rng)
        obs = torch.from_numpy(obs).to(self.device)
        next_obs = torch.from_numpy(next_obs).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        done = torch.from_numpy(done).to(self.device)

        q = self.online(obs)  # [B, N]
        q_taken = q.gather(1, action.unsqueeze(1)).squeeze(1)  # [B]

        with torch.no_grad():
            # Double DQN: select the next action with the online net, value it with the target net.
            next_q_online = self.online(next_obs)
            next_action = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target(next_obs).gather(1, next_action).squeeze(1)
            td_target = reward + cfg.gamma * (1.0 - done) * next_q_target

        loss = F.smooth_l1_loss(q_taken, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % cfg.target_sync_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.online.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)
