# Congestion-Aware Path Selection (DQN)

An RL agent that picks the least-congested of `N` parallel paths to a
destination, based on a simulated per-path congestion state (latency, queue
occupancy, packet loss, utilization, capacity).

## Files

- `config.py` — all hyperparameters and environment settings, including `num_paths`.
- `network_env.py` — `NetworkEnv`, a simulated network with `N` links, each
  with its own capacity/latency/buffer and a time-correlated background
  traffic process (mean-reverting + occasional bursts). The agent's own
  routing choice affects future queue state on that path.
- `agent.py` — `PathScorer` (a shared MLP scoring each path independently, so
  the same weights work for any `N`), a replay buffer, and `DQNAgent`
  (Double DQN, target network, epsilon-greedy).
- `baselines.py` — `RandomPolicy`, `GreedyLatencyPolicy`, `RoundRobinPolicy`.
- `train.py` — trains the agent, logs to `runs/train_log.csv`, saves
  `runs/dqn.pt`.
- `evaluate.py` — runs the trained DQN and all baselines on the same seeded
  episodes and prints a comparison table.
- `observation.py` — converts raw path measurements into the network's
  normalized input format. Shared by the simulator and by real integrations.
- `path_selector.py` — `PathSelector`, the public class to import into
  another project: load a checkpoint once, call `.choose(paths)` per decision.
- `choose_path.py` — minimal example of calling `PathSelector` with
  hand-typed measurements, outside the training/simulation loop.
- `runs/` — saved checkpoints (`dqn.pt`) and training logs (`train_log.csv`).

This is one module of a larger repo — see the root
**[README.md](../README.md)** for how it relates to `scheduler/` and `gan/`.
For how this module works internally and test results, see
**[OVERVIEW.md](OVERVIEW.md)**; for how to use it inside another project, see
**[INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md)**.

## Usage

Run all commands from the **repo root** (`rl/`), as a package, so the
cross-module imports (`path_agent.xxx`) resolve:

```bash
python -m path_agent.network_env   # sanity self-check of the simulator
python -m path_agent.train          # train, writes runs/dqn.pt and runs/train_log.csv
python -m path_agent.evaluate        # compare DQN vs. baselines
python -m path_agent.choose_path      # single-decision example
```

## Changing the number of paths

Edit `num_paths` in `config.py` (or construct `Config(num_paths=6)` in code).
`PathScorer` scores each path with the same small MLP, independent of `N`, so
a checkpoint trained at one `N` can be loaded and run at a different `N`
without any architecture change.

## State and reward

Per path, the observation is a 5-value normalized vector:
`[latency, queue_occupancy, loss_rate, utilization, capacity]`.

Reward for the chosen path each step:

```
reward = -(latency / latency_scale) - loss_penalty * loss_rate
```

so the agent is penalized for both queueing delay and packet loss on the path
it picked, and unaffected by what happens on paths it didn't use that step
(though it still observes them).
