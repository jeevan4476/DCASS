#!/usr/bin/env python3
"""
Generate Synthetic Human Social Media Traffic Dataset for DCASS WGAN Training.

Simulates realistic human posting habits:
- Poisson/Power-law inter-arrival delays (bursty periods followed by long reading pauses)
- Circadian diurnal modulation (sleep/wake daily patterns)
- Multi-channel distributions across social platforms
"""

import json
import random
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "storage" / "data" / "traffic"
OUTPUT_FILE = OUTPUT_DIR / "human_traffic.json"

def generate_human_session(num_items: int = 30, hour_of_day: int = 14) -> dict:
    """Generate a single session of human posting delays and channel choices."""
    delays = []
    channels = []
    
    # Circadian scale factor: nighttime (01:00-06:00) has slower baseline activity
    if 1 <= hour_of_day <= 6:
        base_scale = 12.0
    elif 12 <= hour_of_day <= 14 or 19 <= hour_of_day <= 22:
        base_scale = 2.5  # Peak activity hours
    else:
        base_scale = 5.0
        
    current_channel = random.randint(0, 2)
    
    for _ in range(num_items):
        # 70% chance of bursty browsing (Gamma/Pareto distribution)
        if random.random() < 0.70:
            delay = float(np.random.gamma(shape=1.5, scale=base_scale * 0.8))
        else:
            # 30% chance of longer pause / reading time
            delay = float(np.random.exponential(scale=base_scale * 3.0))
            
        # Clamp to realistic bounds [0.5s, 60.0s]
        delay = float(np.clip(delay, 0.5, 60.0))
        delays.append(round(delay, 2))
        
        # Channel switching probability (80% stay on channel, 20% switch platform)
        if random.random() < 0.20:
            current_channel = random.randint(0, 2)
        channels.append(current_channel)
        
    return {
        "delays": delays,
        "channels": channels,
        "time_of_day": int(hour_of_day)
    }

def main(num_samples: int = 2000):
    print("=" * 70)
    print("Generating Synthetic Human Social Media Traffic Dataset for WGAN-GP")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    sessions = []
    for _ in range(num_samples):
        hour = random.randint(0, 23)
        seq_len = random.randint(15, 50)
        sessions.append(generate_human_session(seq_len, hour))
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)
        
    print(f"✅ Generated {len(sessions):,} human traffic sessions -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
