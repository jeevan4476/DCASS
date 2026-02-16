# src/distribution/noise.py

import random
from typing import List, Tuple

class NoiseController:
    def __init__(
        self,
        seed: int | None = None,
        skip_prob: float = 0.1,
        jitter_range: tuple[int, int] = (-2, 3),
        idle_gap_prob: float = 0.2,
        idle_gap_range: tuple[int, int] = (5, 12)
    ):
        self.random = random.Random(seed)
        self.skip_prob = skip_prob
        self.jitter_range = jitter_range
        self.idle_gap_prob = idle_gap_prob
        self.idle_gap_range = idle_gap_range

    def apply(
        self,
        image_sequence: List[str],
        base_delays: List[int]
    ) -> Tuple[List[str], List[int]]:
        noisy_images = []
        noisy_delays = []

        for image_id, base_delay in zip(image_sequence, base_delays):
            # Random skip
            if self.random.random() < self.skip_prob:
                continue

            # Delay jitter
            jitter = self.random.randint(*self.jitter_range)
            delay = max(0, base_delay + jitter)

            noisy_images.append(image_id)
            noisy_delays.append(delay)

            # Optional idle gap
            if self.random.random() < self.idle_gap_prob:
                gap = self.random.randint(*self.idle_gap_range)
                noisy_images.append(None)   # placeholder
                noisy_delays.append(gap)

        # Remove idle placeholders but keep timing
        final_images = []
        final_delays = []

        for img, d in zip(noisy_images, noisy_delays):
            if img is None:
                time_pass = d
                if final_delays:
                    final_delays[-1] += time_pass
            else:
                final_images.append(img)
                final_delays.append(d)

        return final_images, final_delays
