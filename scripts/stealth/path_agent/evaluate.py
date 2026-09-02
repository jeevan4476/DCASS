import numpy as np

from path_agent.agent import DQNAgent
from path_agent.baselines import GreedyLatencyPolicy, RandomPolicy, RoundRobinPolicy
from path_agent.config import Config
from path_agent.network_env import NetworkEnv


class DQNPolicyWrapper:
    """Wraps a trained DQNAgent so it exposes the same act(obs) interface as baselines."""

    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def act(self, obs: np.ndarray) -> int:
        return self.agent.act(obs, greedy=True)


def run_episode(env: NetworkEnv, policy, seed: int):
    obs = env.reset(seed=seed)
    total_reward = 0.0
    latencies = []
    loss_rates = []
    selections = np.zeros(env.n, dtype=int)

    done = False
    while not done:
        action = policy.act(obs)
        selections[action] += 1
        obs, reward, done, info = env.step(action)
        total_reward += reward
        latencies.append(info["chosen_latency"])
        loss_rates.append(info["chosen_loss_rate"])

    return {
        "reward": total_reward,
        "mean_latency": float(np.mean(latencies)),
        "mean_loss_rate": float(np.mean(loss_rates)),
        "selection_dist": selections / selections.sum(),
    }


def evaluate(cfg: Config | None = None, num_eval_episodes: int = 30, checkpoint_path: str | None = None):
    cfg = cfg or Config()
    checkpoint_path = checkpoint_path or cfg.checkpoint_path

    agent = DQNAgent(cfg)
    agent.load(checkpoint_path)

    policies = {
        "DQN": DQNPolicyWrapper(agent),
        "Random": RandomPolicy(cfg.num_paths, seed=123),
        "GreedyLatency": GreedyLatencyPolicy(),
        "RoundRobin": RoundRobinPolicy(cfg.num_paths),
    }

    # Same seeds for every policy -> identical background-traffic realizations, fair comparison.
    seeds = list(range(10_000, 10_000 + num_eval_episodes))

    results = {name: {"reward": [], "mean_latency": [], "mean_loss_rate": [], "selection_dist": []} for name in policies}

    for name, policy in policies.items():
        env = NetworkEnv(cfg)
        for seed in seeds:
            r = run_episode(env, policy, seed)
            results[name]["reward"].append(r["reward"])
            results[name]["mean_latency"].append(r["mean_latency"])
            results[name]["mean_loss_rate"].append(r["mean_loss_rate"])
            results[name]["selection_dist"].append(r["selection_dist"])

    print(f"\n{'Policy':<14}{'MeanReturn':>12}{'MeanLatency(ms)':>18}{'MeanLossRate':>14}   SelectionDist")
    for name, r in results.items():
        dist = np.mean(r["selection_dist"], axis=0)
        dist_str = "[" + ", ".join(f"{p:.2f}" for p in dist) + "]"
        print(
            f"{name:<14}{np.mean(r['reward']):>12.2f}"
            f"{np.mean(r['mean_latency']):>18.2f}"
            f"{np.mean(r['mean_loss_rate']):>14.4f}   {dist_str}"
        )

    return results


if __name__ == "__main__":
    evaluate()
