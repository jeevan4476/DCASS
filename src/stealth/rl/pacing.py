"""
Canonical pacing for the RL scheduler.

The RL agent chooses a delay and a channel per transmission. That choice
has to mean the same thing in the three places it is used:

  * the training environment, which scores it and rewards against it;
  * the stealth scheduler, which puts it on the wire;
  * the benchmark, which measures the result.

It used to mean three different things. The environment took the agent's
raw delay and scored it directly. The scheduler put
``clip(raw * 0.35, 7.0, 20.0)`` on the wire. The agent was therefore
optimizing a delay sequence that never reached the network, and the
transform was additionally degenerate: because its 7.0 second floor sat
above almost the whole of the agent's action range, 98.3 percent of the
choices it made collapsed onto a single value and the transmitted traffic
was a constant clock.

The window below is the deployment contract. Its lower bound is the
agent's physical minimum rather than a floor derived from the channel
cooldowns, and that distinction matters. The per-channel cooldown is a
*per-channel* constraint, and the pipeline already enforces it with the
action mask, which masks out any channel still cooling down. A global
delay floor on top of that is both unnecessary and destructive: the human
traffic the project is imitating has a median inter-arrival of 5.5
seconds and 58 percent of its delays below 7 seconds, so a 7 second floor
excludes the short-delay mass that makes human traffic look human. The
upper bound keeps a single delay from stretching into a multi-minute
stall.

The agent's action is clamped into the window before the action is
scored, so the value the agent optimizes is the value the receiver sees.
"""

from __future__ import annotations

# The pacing window, in seconds. The lower bound is the shortest
# physically meaningful gap between two transmissions; the upper bound
# bounds the delay so a single step cannot stall the stream.
RL_DELAY_MIN = 0.5
RL_DELAY_MAX = 20.0


def canonical_rl_delay(raw_delay: float) -> float:
    """Map an agent's raw delay onto the deployment window.

    The mapping is monotone and has a strictly positive derivative
    everywhere inside the window, so a variation in the agent's choice
    always produces a variation on the wire. A clip that saturates at its
    floor does not: it maps a wide range of distinct actions onto one
    value, which is how the previous transform erased the agent's timing
    variation entirely.
    """
    return max(RL_DELAY_MIN, min(RL_DELAY_MAX, float(raw_delay)))
