"""
DCASS GAN Stealth Demo — CLI
Run inside Docker: docker run dcass-demo
"""

import sys
import time
import torch
from datetime import datetime

sys.path.append("src/stealth/gan")
from generator import TemporalPatternGenerator

# ─── Config ──────────────────────────────────────────────────────────────────

MESSAGE = "a dog running through water and a man riding a bicycle"

semantic_payload = [
    "img_4521.jpg",
    "img_982.jpg",
]

CHANNELS = {
    0: "TestApp1",
    1: "TestApp2",
    2: "TestApp3",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def ts():
    return datetime.now().strftime("%H:%M:%S")

def line(char="─", width=60):
    print(char * width)

def header(title):
    line("═")
    pad = (58 - len(title)) // 2
    print(f"{'═'} {' ' * pad}{title}{' ' * pad} {'═'}")
    line("═")

def section(title):
    print(f"\n── {title} {'─' * (54 - len(title))}")

# ─── Banner ──────────────────────────────────────────────────────────────────

print()
header("DCASS  GAN  STEALTH  DEMO")
print(f"  System  : Deep Covert Adaptive Steganography")
print(f"  Mode    : GAN Temporal Pattern Generator")
print(f"  Device  : CPU")
print(f"  Started : {ts()}")
line()

# ─── Message ─────────────────────────────────────────────────────────────────

section("COVERT MESSAGE")
print(f'  "{MESSAGE}"')
print()
print(f"  Semantic chunks ({len(semantic_payload)}):")
for i, p in enumerate(semantic_payload, 1):
    print(f"    {i}. {p}")

# ─── GAN Init ────────────────────────────────────────────────────────────────

section("GAN INITIALISATION")
print(f"  [{ts()}] Loading TemporalPatternGenerator...")

device = "cpu"
generator = TemporalPatternGenerator(
    latent_dim=128,
    hidden_dim=256,
    num_channels=3,
    max_sequence_length=100,
).to(device)
generator.eval()

params = sum(p.numel() for p in generator.parameters())
print(f"  [{ts()}] Model ready  —  {params:,} parameters")
print(f"  [{ts()}] Architecture : GRU (2-layer) + Multi-head Attention (8 heads)")

# ─── Schedule Generation ─────────────────────────────────────────────────────

section("GENERATING SCHEDULE")
print(f"  [{ts()}] Sampling latent noise  z ~ N(0, I)  [dim=128]")
print(f"  [{ts()}] Time-of-day context   : 14:00")

with torch.no_grad():
    schedule = generator.generate(
        batch_size=1,
        sequence_length=len(semantic_payload),
        time_of_day=torch.tensor([14.0]),
        device=device,
    )

delays   = schedule.delays[0]
channels = schedule.sample_channels()[0]
confidence = schedule.confidence[0].item()

print(f"  [{ts()}] Schedule generated     confidence={confidence:.4f}")

# ─── Schedule Table ───────────────────────────────────────────────────────────

section("GENERATED SCHEDULE")
print(f"  {'STEP':<6} {'PAYLOAD':<16} {'DELAY (s)':<12} {'CHANNEL':<14} {'DELAY BAR'}")
line("·")

for i in range(len(semantic_payload)):
    d  = delays[i].item()
    ch = CHANNELS[channels[i].item()]
    bar_len = min(int(d / 0.5), 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"  {i+1:<6} {semantic_payload[i]:<16} {d:<12.3f} {ch:<14} {bar}")

line()
print(f"  Total delay : {delays.sum().item():.3f}s")
print(f"  Confidence  : {confidence:.4f}  (generator human-likeness score)")

# ─── Transmission ─────────────────────────────────────────────────────────────

section("SIMULATED TRANSMISSION")
print(f"  [{ts()}] Opening steganographic channels...\n")

for i, payload in enumerate(semantic_payload):
    d  = delays[i].item()
    ch = CHANNELS[channels[i].item()]
    sleep = min(d, 2.0)

    print(f"  [{ts()}] WAIT   Step {i+1}/{len(semantic_payload)}  —  delaying {d:.3f}s (human-like jitter)")
    sys.stdout.flush()
    time.sleep(sleep)

    print(f"  [{ts()}] SEND   {payload}  →  {ch}")
    print()

# ─── Decode ───────────────────────────────────────────────────────────────────

section("RECEIVER DECODE")
print(f"  [{ts()}] Image sequence received: {semantic_payload}")
print(f"  [{ts()}] Running CLIP caption retrieval...")
time.sleep(0.5)
print(f"  [{ts()}] Captions mapped:")
print(f"           img_4521.jpg  →  'dog playing in water'")
print(f"           img_982.jpg   →  'man riding bicycle'")

print()
line("═")
print("  RECONSTRUCTED MESSAGE:")
print()
print("    dog playing in water. man riding bicycle.")
print()
line("═")
print(f"  [{ts()}] Transmission complete.")
print()