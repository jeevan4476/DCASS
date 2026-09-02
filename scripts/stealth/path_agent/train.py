import csv
import os

from path_agent.agent import DQNAgent
from path_agent.config import Config
from path_agent.network_env import NetworkEnv


def train(cfg: Config | None = None):
    cfg = cfg or Config()
    env = NetworkEnv(cfg)
    agent = DQNAgent(cfg)

    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.log_csv_path), exist_ok=True)

    with open(cfg.log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "mean_latency", "mean_loss_rate", "epsilon"])

        for episode in range(cfg.num_episodes):
            obs = env.reset()
            episode_return = 0.0
            latencies = []
            loss_rates = []

            done = False
            while not done:
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                agent.store(obs, action, reward, next_obs, done)
                agent.update()

                obs = next_obs
                episode_return += reward
                latencies.append(info["chosen_latency"])
                loss_rates.append(info["chosen_loss_rate"])

            mean_latency = sum(latencies) / len(latencies)
            mean_loss = sum(loss_rates) / len(loss_rates)
            writer.writerow([episode, episode_return, mean_latency, mean_loss, agent.epsilon()])

            if episode % cfg.log_every == 0:
                f.flush()
                print(
                    f"episode {episode:4d}  return {episode_return:8.2f}  "
                    f"mean_latency {mean_latency:6.2f}ms  mean_loss {mean_loss:.4f}  "
                    f"eps {agent.epsilon():.3f}"
                )

    agent.save(cfg.checkpoint_path)
    print(f"saved checkpoint to {cfg.checkpoint_path}")


if __name__ == "__main__":
    train()
