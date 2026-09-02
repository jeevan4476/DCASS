"""Demonstrates a single routing decision using the public PathSelector API,
independent of the training/simulation loop.

In a real deployment you would replace `paths_raw` with live measurements
(e.g. from ping/traceroute, SNMP counters, or your own probes) instead of
hand-typed numbers.
"""

from path_agent.path_selector import PathSelector


def main():
    selector = PathSelector()  # uses the default checkpoint at path_agent/runs/dqn.pt

    # Example: 4 candidate paths, with made-up "live" measurements.
    paths_raw = [
        {"latency_ms": 15,  "queue": 20,  "buffer": 200, "loss_rate": 0.0,  "arrival_rate": 40, "capacity": 100},  # path 0: quiet
        {"latency_ms": 60,  "queue": 150, "buffer": 200, "loss_rate": 0.02, "arrival_rate": 95, "capacity": 100},  # path 1: congested
        {"latency_ms": 25,  "queue": 40,  "buffer": 250, "loss_rate": 0.0,  "arrival_rate": 60, "capacity": 120},  # path 2: moderate
        {"latency_ms": 90,  "queue": 180, "buffer": 200, "loss_rate": 0.05, "arrival_rate": 98, "capacity": 100},  # path 3: near-saturated
    ]

    chosen = selector.choose(paths_raw)

    print("Path stats:")
    for i, p in enumerate(paths_raw):
        marker = "  <-- chosen" if i == chosen else ""
        print(f"  path {i}: latency={p['latency_ms']}ms queue={p['queue']}/{p['buffer']} "
              f"loss={p['loss_rate']:.2%} util={p['arrival_rate']/p['capacity']:.0%}{marker}")

    print(f"\nAgent chose path {chosen}")


if __name__ == "__main__":
    main()
