# rl

Congestion-aware packet routing: decide which channel to send a packet on,
and (eventually) when, using RL.

## Modules

- **[path_agent/](path_agent/)** — the RL agent itself. A DQN that picks the
  least-congested of `N` parallel paths based on live congestion state
  (latency, queue occupancy, loss, utilization). See
  [path_agent/OVERVIEW.md](path_agent/OVERVIEW.md) for how it works and test
  results, and [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for how to use
  it from another project.
- **[scheduler/](scheduler/)** — given a set of timestamps (e.g. from
  `gan/`) at which a packet must be sent, decides the best channel for each
  one via `path_agent`. Decision-only for now; the actual send mechanism is
  a pluggable callback, still to be finalized.
- **[gan/](gan/)** — a WGAN-GP that generates realistic timestamps (learned
  weekday/time-of-day rhythm) for *when* packets should be sent. See
  [gan/README.md](gan/README.md). Independent of the other two modules; the
  scheduler consumes its output.

## How the pieces fit together

```
gan/  → timestamps (when to send)
              │
              ▼
scheduler/  → for each timestamp, asks path_agent for the best channel
              │
              ▼
path_agent/ → PathSelector.choose(paths) → channel index
```

`gan/` and `path_agent/` don't depend on each other; `scheduler/` is the
glue between them. Each module can be developed, tested, and (eventually)
deployed independently.

Run the whole pipeline interactively with:

```bash
python run_pipeline.py
```

It asks for three things — number of packets, number of channels, and a
date — then generates that many real timestamps from the trained GAN
(`gan/gan_ckpt.pt`) on that date, and for each one asks `path_agent` (via a
simulated congestion source, for now) which channel to send on. If you
already have a checkpoint trained specifically for that channel count
(`path_agent/runs/dqn_n{N}.pt`), it's used automatically; otherwise it falls
back to the default checkpoint, which still works at any channel count.

`python -m scheduler.demo_scheduler` runs the same pipeline non-interactively
with fixed defaults, useful for quick smoke-testing.

Congestion is currently simulated and sending is not yet wired to anything
real — see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) section 8 for what's
left to connect a real congestion source and send mechanism.

## Setup

```bash
pip install -r requirements.txt   # torch, numpy, pandas
```

Run everything from this root directory as a package (`python -m <module>.<file>`)
so the cross-module imports (`path_agent.xxx`) resolve correctly — see each
module's own docs for exact commands.

This project was developed against the venv at
`C:\Users\kappa\OneDrive\sem5\ML\vir`.
