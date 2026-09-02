# Integration Guide — Using This Agent In Another Project

This describes how to plug the trained path-selection agent into a real
project as a routing decision component, not just run it inside this
repo's own simulator.

This repo is organized as one folder per module: `path_agent/` (this
agent), `scheduler/` (offline decision scheduling), and `gan/` (a separate
timestamp-generating GAN — see its own `gan/README.md`). This guide covers
`path_agent/` and `scheduler/`.

## 1. What you actually need to ship

You do **not** need the training code, the simulator, or the baselines in
the consuming project. The runtime dependency surface is small — everything
lives under `path_agent/`:

| file | purpose |
|---|---|
| `path_agent/config.py` | `Config` dataclass (only used to construct the network with the right shapes) |
| `path_agent/agent.py` | `PathScorer` (the network) + `DQNAgent` (loads weights, does inference) |
| `path_agent/observation.py` | Converts your raw measurements into the network's input format |
| `path_agent/path_selector.py` | The one class you actually import: `PathSelector` |
| `path_agent/runs/dqn.pt` (or whichever checkpoint you trained) | The trained weights |

Copy the whole `path_agent/` folder into the consuming project (keeping it
as a package — the files import each other as `path_agent.xxx`), or vendor
this repo as a git submodule / `pip install -e .` it — either works, there's
no packaging magic required since it's plain Python + torch.

Install once: `pip install -r requirements.txt` (just `torch` and `numpy`).

## 2. The integration point: `PathSelector`

```python
from path_agent.path_selector import PathSelector

# Load once, at startup — loading the checkpoint has some cost,
# calling .choose() afterward is a cheap forward pass. Uses
# path_agent/runs/dqn.pt by default.
selector = PathSelector()

# On every routing decision:
chosen_index = selector.choose(paths)
```

`paths` is a list of dicts, one per candidate path, **in the order you want
them indexed** (the returned int refers to that position). Each dict needs:

```python
{
    "latency_ms":   float,   # measured round-trip or one-way latency
    "queue":        float,   # current queue occupancy on that path
    "buffer":       float,   # that path's max queue size
    "loss_rate":    float,   # recent packet loss fraction, 0..1
    "arrival_rate": float,   # current traffic rate on the path
    "capacity":     float,   # the path's bandwidth/capacity, same units as arrival_rate
}
```

`selector.choose(paths)` returns an `int` — the index into your `paths`
list of the path to send traffic on next. That's the entire contract.

Any number of paths ≥ 2 works, and it doesn't have to match whatever `N`
the checkpoint was originally trained with — the network scores each path
independently (see `OVERVIEW.md` section 2.4 for why).

## 3. Where do the numbers in `paths` come from?

This is the part specific to your project — plug in whatever congestion
signal you already have or can add:

- **Active probing**: periodic ping/traceroute per path for latency and loss;
  a lightweight iperf-style probe for available capacity.
- **Passive/application-level**: RTT sampled from your own traffic (e.g.
  TCP RTT via socket stats, or app-layer request/response timing) plus a
  rolling packet-loss counter.
- **Infrastructure metrics**: SNMP/router counters for queue depth,
  interface utilization, and configured link capacity if you control the
  network devices.
- **Overlay/SD-WAN style**: many SD-WAN and multipath libraries already
  expose exactly these fields (latency, loss, jitter, utilization) per
  tunnel/path — if you have one of those, this is a drop-in reuse of data
  you're already collecting.

Whatever the source, keep the units consistent with what you feed in
(e.g. always ms for latency, same unit for `queue`/`buffer`, same unit for
`arrival_rate`/`capacity`) — the exact units don't matter since everything
is normalized as a ratio, but they must be internally consistent per call.

## 4. How often to call it

`selector.choose()` is a single forward pass through a small MLP — call it
as often as you have fresh measurements, e.g.:
- Per outgoing flow/connection (route each new flow once, keep it pinned
  to that path for its lifetime), or
- On a fixed timer (e.g. every 1-5 seconds) re-evaluating which path new
  traffic should default to.

Avoid calling it per-packet at wire speed unless you've benchmarked that
your latency budget allows it — it's fast, but there's no need for that
granularity since congestion doesn't change that quickly.

## 5. Fallback / safety

Wrap the call so a bug or missing measurement never blocks routing
entirely:

```python
from path_agent.baselines import GreedyLatencyPolicy
from path_agent.observation import build_observation

fallback = GreedyLatencyPolicy()

def pick_path(paths_raw):
    try:
        return selector.choose(paths_raw)
    except Exception:
        # fall back to "lowest reported latency" if the agent call fails
        return fallback.act(build_observation(paths_raw))
```

`GreedyLatencyPolicy` needs no trained weights and no state, so it's a safe
degrade path.

## 6. Retraining / adapting to your real traffic

The shipped checkpoint (`path_agent/runs/dqn.pt`) was trained entirely on
the simulated traffic patterns in `path_agent/network_env.py`
(mean-reverting background load + occasional bursts). Before trusting it in
production:

1. **Validate first**: log `paths_raw` from production for a while, replay
   those measurements through `selector.choose()` offline, and sanity-check
   the choices against what a human/ops team would pick.
2. **If behavior looks off**, the fastest fix is usually adjusting
   `path_agent/config.py`'s traffic-dynamics parameters
   (`background_mean_frac`, `burst_prob`, `burst_magnitude_frac`,
   `capacity_range`, etc.) to better match your real network's statistics,
   then rerun `python -m path_agent.train` (from the repo root).
3. **For a closer match to reality**, replace `NetworkEnv` with a version
   whose `step()` replays real recorded traffic traces (the observation
   format and reward function can stay the same — see `OVERVIEW.md`
   section 2.2-2.3) and retrain against that.
4. Re-run `python -m path_agent.evaluate` after any retraining and confirm
   the new checkpoint still beats `GreedyLatencyPolicy` before deploying it.

## 7. Exposing it to non-Python callers

If the consuming project isn't Python, wrap `PathSelector` in a tiny HTTP
service, e.g. with FastAPI:

```python
from fastapi import FastAPI
from path_agent.path_selector import PathSelector

app = FastAPI()
selector = PathSelector()

@app.post("/choose-path")
def choose_path(paths: list[dict]):
    return {"chosen_index": selector.choose(paths)}
```

Then any language just POSTs its list of path measurements and gets an
index back.

## 8. Scheduling sends at externally-generated timestamps (e.g. the GAN)

The `gan/` module (see `gan/README.md`) generates *when* to send a packet;
this agent decides *which channel* to send it on at each of those moments.
The `scheduler/` package (`scheduler/scheduler.py`) is the glue:

```python
from path_agent.config import Config
from path_agent.path_selector import PathSelector
from scheduler import plan_sends
from gan.gan import generate_timestamps   # the real GAN

cfg = Config()
selector = PathSelector()

timestamps = generate_timestamps(20, start="2026-09-15", end="2026-09-16", ckpt="gan/gan_ckpt.pt")

schedule = plan_sends(timestamps, selector, congestion_provider=my_congestion_source)
# schedule: list[ScheduledSend], each with .timestamp, .channel, .path_stats
```

`plan_sends` is decision-only — it does **not** send anything. Pass a
`send_fn` callback once the send mechanism is decided:

```python
def send_fn(entry):
    real_send(entry.channel, ...)   # wire this up when ready

plan_sends(timestamps, selector, my_congestion_source, send_fn=send_fn)
```

`congestion_provider` is either:
- a plain function `f(timestamp) -> paths_raw` for a real/live measurement
  source (same `paths_raw` format as section 3), or
- an object with `.observe(timestamp) -> paths_raw` and optional
  `.commit(channel)`, for a stateful source that needs to know which
  channel was picked (e.g. a simulator, so subsequent timestamps see
  updated congestion from that choice).

**Testing before a real congestion source or send mechanism exists**: the
`scheduler` package includes `SimulatedCongestionProvider`, which drives
`NetworkEnv` under the hood so you can validate the whole pipeline
(timestamps → congestion → chosen channel) today:

```python
from scheduler import SimulatedCongestionProvider
congestion = SimulatedCongestionProvider(config=cfg, seed=1)
schedule = plan_sends(timestamps, selector, congestion)
```

`scheduler/demo_scheduler.py` runs the full real pipeline — GAN timestamps
→ scheduler → chosen channel — via `python -m scheduler.demo_scheduler`
from the repo root. It uses the real GAN (`gan/gan.py`) automatically when
`gan/gan_ckpt.pt` exists, falling back to a lightweight stand-in
(`np.cumsum` of random gaps) only if it doesn't, so the module still runs
standalone before a GAN checkpoint is trained.

**Current limitation**: this is offline planning. Each timestamp advances
the simulated congestion by one fixed step regardless of the actual time
gap between timestamps, since the simulator doesn't yet model arbitrary
elapsed real time between decisions. When you move to real-time operation,
swap `SimulatedCongestionProvider` for a live measurement source (a plain
`f(timestamp) -> paths_raw` function reading current conditions at call
time) — `plan_sends` itself doesn't need to change, and you'd typically
sleep until each timestamp before calling it rather than planning the whole
batch up front.

## 9. Checklist before go-live

- [ ] `python -m path_agent.network_env` self-check passes (confirms your
      local install of the code is intact; run from the repo root).
- [ ] `python -m path_agent.evaluate` shows the checkpoint you intend to
      ship beating `GreedyLatencyPolicy` and `Random`.
- [ ] Confirmed the field units your measurement pipeline produces match
      what `observation.py` expects (see section 3).
- [ ] Fallback path (section 5) is wired in.
- [ ] Decision cadence (section 4) matches your latency/traffic
      requirements.
