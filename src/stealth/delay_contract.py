"""The canonical mapping between raw generator output and delay seconds.

This module is the single definition of the deployment contract for the
generator's delay head. Both the scheduler, which deploys the traffic,
and the generator, which produces it, read it from here, so the two
cannot drift apart.

Why the contract is shaped this way
-----------------------------------
The generator's delay head ends in a Softplus, so its raw output is
non-negative and unbounded. The canonical mapping is
``(raw - 0.5) * 1500 + 1.5`` clipped to ``[0.5, 30.0]`` seconds, so only a
narrow band of raw values survives the clip:

    raw in [0.49933, 0.51900]

That band is 1.97 percent of the Softplus output's range. A generator
whose output wanders outside it has its delays flattened against the
clip, and the gradient that would bring it back is identically zero
there, because a clipped value has no gradient. A freshly initialised
generator had 100 percent of its outputs at the upper clip and 100
percent of its gradient zero.

The generator therefore emits its raw value inside the band, using a
bounded transform of the Softplus output. The mapping is then never
clipped, the gradient always reaches the head, and the deployed delay
range is the full 0.5 to 30 seconds.

The measured cause is in
``docs/benchmarking/C3_GENERATOR_ADVERSARY.md``.
"""

from __future__ import annotations

import numpy as np
import torch

#: The scale and offset of the canonical mapping.
GAN_DELAY_SCALE = 1500.0
GAN_DELAY_OFFSET = 1.5

#: The deployment window, in seconds.
GAN_DELAY_MIN = 0.5
GAN_DELAY_MAX = 30.0

#: The raw values the mapping can turn into the deployment window
#: without clipping either end.
GAN_RAW_MIN = 0.5 + (GAN_DELAY_MIN - GAN_DELAY_OFFSET) / GAN_DELAY_SCALE
GAN_RAW_MAX = 0.5 + (GAN_DELAY_MAX - GAN_DELAY_OFFSET) / GAN_DELAY_SCALE

#: The width of that band, in raw units.
GAN_RAW_BAND = GAN_RAW_MAX - GAN_RAW_MIN


def canonical_delay_mapping(raw) -> np.ndarray:
    """Map raw generator outputs to inter-item delays in seconds.

    ``(raw - 0.5) * 1500 + 1.5``, clipped to ``[0.5, 30.0]`` seconds.
    """
    raw_arr = np.asarray(raw, dtype=np.float64)
    mapped = (raw_arr - 0.5) * GAN_DELAY_SCALE + GAN_DELAY_OFFSET
    return np.clip(mapped, GAN_DELAY_MIN, GAN_DELAY_MAX)


def canonical_delay_mapping_torch(raw: torch.Tensor) -> torch.Tensor:
    """The canonical mapping, differentiably.

    The scheduler's mapping is a numpy clip, so no gradient survives it.
    The generator has to be trained through this mapping, so the training
    path needs a version that keeps the gradient alive.

    The clip is still applied, because the mapping is the contract, but
    the generator's output band is chosen so the clip is not reached.
    """
    mapped = (raw - 0.5) * GAN_DELAY_SCALE + GAN_DELAY_OFFSET
    return torch.clamp(mapped, GAN_DELAY_MIN, GAN_DELAY_MAX)


def raw_delay_band_transform(head_output: torch.Tensor) -> torch.Tensor:
    """Map a Softplus delay-head output into the usable raw band.

    The head's output is non-negative, so it is folded into ``[0, 1)``
    before being placed in the band. The fold is a saturating
    transform, so the gradient stays live over the head's whole output
    range, unlike a linear scale that would push everything into the
    clip.
    """
    normalised = head_output / (1.0 + head_output)
    return GAN_RAW_MIN + GAN_RAW_BAND * normalised
