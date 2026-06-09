# DCASS Project - Incomplete Tasks & Implementation PRD

**Document Version:** 1.0  
**Date:** April 6, 2026  
**Project:** DCASS (Dynamic Context-Aware Semantic Steganography)  
**Overall Completion:** ~85%

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Section A: GAN Training Implementation](#2-section-a-gan-training-implementation)
3. [Section B: RL Agent Training Implementation](#3-section-b-rl-agent-training-implementation)
4. [Section C: Decoding Logic Integration](#4-section-c-decoding-logic-integration)
5. [Section D: Frontend Incomplete Features](#5-section-d-frontend-incomplete-features)
6. [Section E: Testing & Quality Assurance](#6-section-e-testing--quality-assurance)
7. [Section F: Production Readiness](#7-section-f-production-readiness)
8. [Appendix: File References](#8-appendix-file-references)

---

## 1. Executive Summary

### Current State
DCASS is a zero-modification steganography system that encodes messages by curating semantically aligned, naturally occurring media. The core encoding/decoding engine is **100% complete**, but the AI-powered stealth components (GAN and RL) are **implemented but not trained**.

### What's Working
- Semantic encoding/decoding pipeline
- FAISS-based multi-modal corpus indexing (images, text, audio)
- Static NoiseController for human-like timing (fallback mode)
- CLI and API interfaces
- Docker simulation environment

### What Needs Completion
| Priority | Component | Status | Effort |
|----------|-----------|--------|--------|
| HIGH | GAN Model Training | Code ready, not trained | 2-4 hours |
| HIGH | RL Agent Training | Code ready, not trained | 4-8 hours |
| HIGH | Decoding Integration | Partial implementation | 2-3 hours |
| MEDIUM | Frontend /decode Page | Not started | 4-6 hours |
| MEDIUM | Wire View File Watcher | Partial | 2-3 hours |
| LOW | Unit Tests | Minimal coverage | 8-12 hours |
| LOW | Production Docker | Dev mode only | 4-6 hours |

---

## 2. Section A: GAN Training Implementation

### 2.1 Overview

The GAN (Generative Adversarial Network) learns to generate human-like transmission timing patterns to evade Deep Packet Inspection (DPI).

**Architecture:**
```
Generator (TemporalPatternGenerator)
    Latent Noise + Time Embedding -> GRU -> Self-Attention -> Output Heads
    - Delay Head: Inter-transmission delays (Softplus activation)
    - Channel Head: Channel selection logits
    - Confidence Head: Generator confidence score

Discriminator (DeepPacketInspectionWarden)
    Delays + Channels -> BiLSTM -> Transformer -> Bot Probability
```

### 2.2 Files Involved

| File | Purpose | Lines |
|------|---------|-------|
| `src/stealth/gan/generator.py` | TemporalPatternGenerator network | 379 |
| `src/stealth/gan/trainer.py` | GANTrainer with WGAN-GP support | 497 |
| `src/analysis/adversarial/warden.py` | DeepPacketInspectionWarden | 494 |
| `scripts/training/train_gan.py` | Training entry point | 88 |
| `scripts/training/generate_traffic_data.py` | Synthetic data generator | ~100 |

### 2.3 Step-by-Step Training Guide

#### Step 1: Generate Training Data

Human traffic data must be generated before training. The data should simulate realistic social media posting patterns.

```bash
# Using Docker (Recommended)
docker compose --profile training run dcass-gen-traffic

# Or locally
python scripts/training/generate_traffic_data.py \
    --num-sessions 2000 \
    --num-channels 3 \
    --output data/behavioral/human_traffic.json
```

**Expected Output Format** (`data/behavioral/human_traffic.json`):
```json
[
    {
        "delays": [5.2, 3.1, 12.4, 8.7, ...],
        "channels": [0, 1, 0, 2, 1, ...],
        "time_of_day": 14
    },
    ...
]
```

**Data Requirements:**
- Minimum 1000 sessions (2000+ recommended)
- Variable sequence lengths (10-100 items)
- Delays: Exponential distribution (Poisson-like burstiness)
- Channels: Random switching with session-level preferences
- Time of day: 0-23 hour distribution

#### Step 2: Configure Training

Edit training parameters as needed:

```python
# In scripts/training/train_gan.py or via CLI arguments
config = TrainingConfig(
    latent_dim=128,           # Generator noise dimension
    hidden_dim=256,           # Network hidden size
    num_channels=3,           # Distribution channels
    max_sequence_length=50,   # Max items per sequence
    batch_size=32,            # Training batch size
    num_epochs=50,            # Training iterations (50-100 recommended)
    generator_lr=1e-4,        # Generator learning rate
    warden_lr=2e-4,           # Discriminator learning rate (2x generator)
    warden_steps=5,           # Discriminator updates per generator update
    use_gradient_penalty=False,  # Enable for WGAN-GP (more stable)
    lambda_gp=10.0,           # Gradient penalty coefficient
    device="cpu",             # Use "cuda" if GPU available
)
```

#### Step 3: Run Training

```bash
# Using Docker (Recommended for isolation)
docker compose --profile training run dcass-train-gan

# Or with custom parameters
docker compose --profile training run dcass-train-gan \
    python -u scripts/training/train_gan.py --epochs 100 --batch-size 64

# Or locally
python scripts/training/train_gan.py \
    --data data/behavioral/human_traffic.json \
    --epochs 50 \
    --batch-size 32 \
    --device cpu \
    --checkpoint-dir models/gan

# For more stable training (WGAN-GP)
python scripts/training/train_gan.py --wgan-gp --epochs 100
```

#### Step 4: Monitor Training

```bash
# Start TensorBoard
docker compose --profile monitoring up tensorboard

# Access at http://localhost:6006
```

**Key Metrics to Watch:**
- `generator_loss`: Should decrease over time
- `warden_loss`: Should stabilize (not collapse to 0)
- `fake_bot_prob`: Should decrease (generator fooling warden)
- `real_bot_prob`: Should remain low (~0.1-0.3)

**Healthy Training Signs:**
- Generator loss: 0.5 -> 0.3 over 50 epochs
- Fake bot probability: 0.9 -> 0.3 (generator becoming stealthier)
- Warden loss oscillating but not collapsing

#### Step 5: Validate Training

```python
# Test the trained generator
import torch
from src.stealth.gan.generator import TemporalPatternGenerator

generator = TemporalPatternGenerator(num_channels=3)
checkpoint = torch.load("models/gan/final.pt")
generator.load_state_dict(checkpoint["generator_state"])
generator.eval()

# Generate a schedule
schedule = generator.generate(
    batch_size=1,
    sequence_length=20,
    time_of_day=torch.tensor([14.0])  # 2 PM
)

print(f"Delays: {schedule.delays[0].tolist()}")
print(f"Channels: {schedule.sample_channels()[0].tolist()}")
print(f"Confidence: {schedule.confidence.item():.3f}")
```

#### Step 6: Output Files

After training, these files should exist:
```
models/gan/
    epoch_000.pt    # Checkpoint at epoch 0
    epoch_001.pt    # Checkpoint at epoch 1
    ...
    epoch_049.pt    # Checkpoint at epoch 49
    final.pt        # Final trained model
```

**Checkpoint Structure:**
```python
{
    "epoch": int,
    "global_step": int,
    "generator_state": dict,      # Generator weights
    "warden_state": dict,         # Warden weights
    "generator_optimizer": dict,
    "warden_optimizer": dict,
    "config": TrainingConfig,
    "metrics_history": list
}
```

### 2.4 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Generator loss = 0 | Mode collapse | Reduce warden_steps, use WGAN-GP |
| Warden loss = 0 | Warden too strong | Increase generator_lr, reduce warden_lr |
| NaN losses | Learning rate too high | Reduce learning rates by 10x |
| Training slow | No GPU | Use device="cuda" or reduce epochs |
| Out of memory | Batch too large | Reduce batch_size to 16 or 8 |

---

## 3. Section B: RL Agent Training Implementation

### 3.1 Overview

The RL (Reinforcement Learning) agent learns optimal transmission scheduling using Proximal Policy Optimization (PPO). It balances throughput (fast transmission) against stealth (avoiding detection).

**Architecture:**
```
Actor-Critic Network (PPOAgent)
    State -> Shared MLP -> Actor Head (delay distribution, channel logits)
                       -> Critic Head (value estimate V(s))

Environment (StealthEnvironment)
    - State: Queue size, time of day, channel states, history features
    - Action: (delay: continuous, channel: discrete)
    - Reward: Throughput - lambda * Warden_Score
```

### 3.2 Files Involved

| File | Purpose | Lines |
|------|---------|-------|
| `src/stealth/rl/agent.py` | PPOAgent with ActorCritic network | 579 |
| `src/stealth/rl/environment.py` | StealthEnvironment (Gym-style) | 492 |
| `scripts/training/train_rl.py` | Training entry point | 84 |

### 3.3 Step-by-Step Training Guide

#### Step 1: Prerequisites

**Option A: Train with Fresh Warden** (Easier, less effective)
```bash
python scripts/training/train_rl.py --episodes 1000
```

**Option B: Train with Pre-trained Warden** (Recommended)
```bash
# First, train the GAN (Section A)
# Then use the trained Warden
python scripts/training/train_rl.py \
    --warden-checkpoint models/gan/final.pt \
    --episodes 2000
```

#### Step 2: Configure Training

```python
# PPO Configuration
config = PPOConfig(
    state_dim=16,             # Environment state dimension
    hidden_dim=256,           # Network hidden size
    learning_rate=3e-4,       # Adam learning rate
    gamma=0.99,               # Discount factor
    epsilon_clip=0.2,         # PPO clipping parameter
    value_loss_coef=0.5,      # Value loss weight
    entropy_coef=0.01,        # Entropy bonus (exploration)
    max_grad_norm=0.5,        # Gradient clipping
    num_epochs=4,             # PPO epochs per update
    batch_size=64,            # Mini-batch size
    device="cpu"              # Use "cuda" if available
)

# Environment Configuration
env = StealthEnvironment(
    num_channels=3,           # Distribution channels
    warden=trained_warden,    # Pre-trained Warden (or fresh)
    lambda_stealth=100.0,     # Stealth penalty coefficient
    max_episode_time=3600.0,  # Max episode duration (1 hour)
    warden_window_size=20     # Recent transmissions for evaluation
)
```

#### Step 3: Run Training

```bash
# Using Docker (Recommended)
docker compose --profile training run dcass-train-rl

# With pre-trained Warden
docker compose --profile training run dcass-train-rl \
    python -u scripts/training/train_rl.py \
        --warden-checkpoint /app/storage/models/gan/final.pt \
        --episodes 2000

# Locally
python scripts/training/train_rl.py \
    --episodes 1000 \
    --num-channels 3 \
    --lambda-stealth 100.0 \
    --lr 3e-4 \
    --device cpu \
    --checkpoint-dir models/rl

# With trained Warden
python scripts/training/train_rl.py \
    --warden-checkpoint models/gan/final.pt \
    --episodes 2000
```

#### Step 4: Monitor Training

**Console Output:**
```
Episode 100/1000 | Avg Reward: 45.23 | Avg Length: 18.2 | Warden Score: 0.312 | Policy Loss: 0.0234
Episode 200/1000 | Avg Reward: 62.41 | Avg Length: 19.8 | Warden Score: 0.245 | Policy Loss: 0.0189
...
```

**Key Metrics:**
- `Avg Reward`: Should increase over time
- `Warden Score`: Should decrease (more stealthy)
- `Policy Loss`: Should decrease and stabilize
- `Avg Length`: Episode completion (higher = better throughput)

**Healthy Training Signs:**
- Reward: 20 -> 80+ over 1000 episodes
- Warden Score: 0.5 -> 0.2 (becoming stealthier)
- Stable policy loss (not oscillating wildly)

#### Step 5: Validate Training

```python
import torch
import numpy as np
from src.stealth.rl.agent import PPOAgent, PPOConfig
from src.stealth.rl.environment import StealthEnvironment
from src.analysis.adversarial.warden import DeepPacketInspectionWarden

# Load trained agent
warden = DeepPacketInspectionWarden(num_channels=3)
warden.eval()

env = StealthEnvironment(num_channels=3, warden=warden)
config = PPOConfig(state_dim=env.state_dim, device="cpu")
agent = PPOAgent(env, config)

# Load checkpoint
agent.load("models/rl/ppo_agent_final.pt")

# Test episode
media_sequence = [f"media_{i:03d}" for i in range(20)]
state = env.reset(media_sequence)
total_reward = 0

for step in range(100):
    action, _, _ = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    
    print(f"Step {step}: delay={action['delay']:.1f}s, channel={action['channel']}, reward={reward:.2f}")
    
    if done:
        break
    state = next_state

print(f"\nTotal Reward: {total_reward:.2f}")
print(f"Final Warden Score: {env.get_warden_score():.3f}")
```

#### Step 6: Output Files

```
models/rl/
    ppo_agent_final.pt    # Final trained agent
```

**Checkpoint Structure:**
```python
{
    "actor_critic_state": dict,   # Actor-Critic weights
    "optimizer_state": dict,
    "config": PPOConfig,
    "episode_rewards": list,
    "episode_lengths": list,
    "warden_scores": list
}
```

### 3.4 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Reward not improving | lambda_stealth too high | Reduce to 50.0 or lower |
| Agent always waits | High warden penalty | Reduce lambda_stealth |
| Agent too aggressive | Low warden penalty | Increase lambda_stealth |
| Unstable training | Learning rate | Reduce lr to 1e-4 |
| Poor exploration | Low entropy coef | Increase entropy_coef to 0.05 |

### 3.5 Recommended Training Pipeline

```bash
# Full training pipeline (recommended order)

# 1. Generate traffic data
docker compose --profile training run dcass-gen-traffic

# 2. Train GAN (50-100 epochs)
docker compose --profile training run dcass-train-gan

# 3. Train RL with trained Warden (1000-2000 episodes)
RL_EPISODES=2000 docker compose --profile training run dcass-train-rl

# 4. Verify models exist
ls -la models/gan/final.pt models/rl/ppo_agent_final.pt

# 5. Test with sender/receiver simulation
docker compose up
```

---

## 4. Section C: Decoding Logic Integration

### 4.1 Overview

The SemanticDecoder is **implemented** but needs **integration** with the simulation pipeline. Currently, decoding works but the receiver daemon doesn't fully utilize it.

### 4.2 Current Implementation Status

| Component | Status | File |
|-----------|--------|------|
| SemanticDecoder class | Complete | `src/engine/decoder.py` |
| DecodingResult dataclass | Complete | `src/engine/decoder.py` |
| Corpus lookup (get_by_id) | Complete | `src/corpus/index/unified_index.py` |
| Receiver daemon | Partial | `scripts/runtime/run_receiver.py` |

### 4.3 What Needs to be Done

#### Task 1: Enable Decoder in Receiver

**Current Code** (`scripts/runtime/run_receiver.py`, lines 224-235):
```python
# Decode message (if decoder is available)
try:
    # Note: In simulation mode without actual indices,
    # we would just log the sequence
    print(f"[Receiver] Media sequence: {media_ids}")

    # Uncomment when indices are available:
    # decoded_message = self.decoder.decode(media_ids)
    # print(f"[Receiver] Decoded message: {decoded_message}")
    # self.decoded_messages.append(decoded_message)
```

**Required Changes:**
```python
# Replace with:
async def reassemble_and_decode(self):
    """Reassemble buffered packets and decode message."""
    print(f"[Receiver] Silence threshold reached. Reassembling {len(self.buffer)} packets...")

    media_ids = self.buffer.reassemble()
    if not media_ids:
        print("[Receiver] No packets to reassemble")
        return

    print(f"[Receiver] Reassembled sequence: {media_ids}")

    # Decode if decoder is loaded
    if self._decoder_loaded:
        try:
            result = self.decoder.decode(media_ids)
            
            print(f"[Receiver] Verification rate: {result.verification_rate:.1%}")
            print(f"[Receiver] Decoded content:")
            for item in result.decoded:
                status = "OK" if item.verified else "UNVERIFIED"
                print(f"  [{status}] {item.modality}:{item.media_id} -> \"{item.content[:50]}...\"")
            
            print(f"[Receiver] Reconstructed meaning: \"{result.reconstructed_meaning}\"")
            self.decoded_messages.append(result.reconstructed_meaning)
            
        except Exception as e:
            print(f"[Receiver] Decoding error: {e}")
    else:
        print(f"[Receiver] Decoder not loaded - raw sequence: {media_ids}")
```

#### Task 2: Add Index Loading to Receiver

**Add to ReceiverDaemon.__init__:**
```python
def __init__(self, ..., load_indices: bool = False):
    ...
    self._should_load_indices = load_indices
    
    if load_indices:
        self._load_decoder()

def _load_decoder(self):
    """Load the semantic decoder with indices."""
    print("[Receiver] Loading semantic decoder and indices...")
    try:
        self._decoder = SemanticDecoder(base_path=Path("data/indices"))
        status = self._decoder.load()
        self._decoder_loaded = any(status.values())
        
        if self._decoder_loaded:
            print(f"[Receiver] Decoder loaded: {status}")
        else:
            print("[Receiver] Warning: No indices loaded, decoding disabled")
    except Exception as e:
        print(f"[Receiver] Failed to load decoder: {e}")
        self._decoder_loaded = False
```

#### Task 3: Add CLI Flag

**Update main() in run_receiver.py:**
```python
parser.add_argument(
    "--decode",
    action="store_true",
    help="Enable full decoding (requires corpus indices)"
)

# In daemon creation:
receiver = ReceiverDaemon(
    watch_directory=Path(args.watch),
    silence_threshold=args.timeout,
    poll_interval=args.poll_interval,
    load_indices=args.decode  # New parameter
)
```

### 4.4 End-to-End Decoding Test

```python
# test_decoding.py
from pathlib import Path
from src.engine.encoder import SemanticEncoder
from src.engine.decoder import SemanticDecoder

# Initialize
encoder = SemanticEncoder(base_path=Path("data/indices"))
decoder = SemanticDecoder(base_path=Path("data/indices"))

# Load indices
encoder.load()
decoder.load()

# Encode a message
message = "Meet at the cafe at noon"
encoding_result = encoder.encode(message, num_results=5)
media_ids = [item.media_id for item in encoding_result.items]
print(f"Encoded to: {media_ids}")

# Decode back
decoding_result = decoder.decode(media_ids)
print(f"Decoded: {decoding_result.reconstructed_meaning}")
print(f"Verification: {decoding_result.verification_rate:.1%}")
```

---

## 5. Section D: Frontend Incomplete Features

### 5.1 Overview

The Next.js frontend is **85% complete**. Two main features are missing:

| Feature | Page | Status | Priority |
|---------|------|--------|----------|
| Decode Dashboard | `/decode` | Not started | HIGH |
| Wire View File Watcher | `/wire` | Partial (80%) | MEDIUM |

### 5.2 Feature 1: Decode Dashboard (`/decode`)

**Purpose:** Bob's interface to view received packets, reassembly status, and decoded messages.

#### Requirements

1. **Packet List View**
   - Display received packets in real-time
   - Show: media_id, channel, sequence_number, timestamp
   - Color-code by channel

2. **Reassembly Status**
   - Show buffer contents
   - Display silence countdown timer
   - "Reassemble Now" button

3. **Decoded Message Display**
   - Show reconstructed semantic meaning
   - Display verification status per item
   - Highlight unverified items

#### Implementation Steps

**Step 1: Create page file**
```
frontend/src/app/decode/page.tsx
```

**Step 2: API endpoint needed**
```typescript
// Required API endpoint: GET /api/receiver/status
interface ReceiverStatus {
  buffer: {
    packets: Array<{
      media_id: string;
      channel_id: number;
      sequence_number: number;
      timestamp: number;
    }>;
    silence_remaining: number;
  };
  decoded_messages: Array<{
    media_ids: string[];
    reconstructed: string;
    verification_rate: number;
    timestamp: number;
  }>;
}
```

**Step 3: Component structure**
```typescript
// frontend/src/app/decode/page.tsx
'use client';

import { useState, useEffect } from 'react';

interface Packet {
  media_id: string;
  channel_id: number;
  sequence_number: number;
  timestamp: number;
}

interface DecodedMessage {
  media_ids: string[];
  reconstructed: string;
  verification_rate: number;
  timestamp: number;
}

export default function DecodePage() {
  const [packets, setPackets] = useState<Packet[]>([]);
  const [messages, setMessages] = useState<DecodedMessage[]>([]);
  const [silenceRemaining, setSilenceRemaining] = useState(0);

  // Poll for updates
  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch('/api/receiver/status');
      const data = await res.json();
      setPackets(data.buffer.packets);
      setMessages(data.decoded_messages);
      setSilenceRemaining(data.buffer.silence_remaining);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Decode Dashboard (Bob)</h1>
      
      {/* Buffer Status */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Reassembly Buffer</h2>
        <div className="bg-gray-100 p-4 rounded">
          <p>Packets in buffer: {packets.length}</p>
          <p>Silence remaining: {silenceRemaining.toFixed(1)}s</p>
        </div>
        
        {/* Packet list */}
        <div className="mt-4 space-y-2">
          {packets.map((p, i) => (
            <div key={i} className={`p-2 rounded channel-${p.channel_id}`}>
              #{p.sequence_number} | {p.media_id} | Ch{p.channel_id}
            </div>
          ))}
        </div>
      </section>

      {/* Decoded Messages */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Decoded Messages</h2>
        {messages.map((msg, i) => (
          <div key={i} className="border p-4 mb-4 rounded">
            <p className="font-mono text-sm text-gray-500">
              {msg.media_ids.join(' -> ')}
            </p>
            <p className="text-lg mt-2">{msg.reconstructed}</p>
            <p className="text-sm text-green-600">
              Verified: {(msg.verification_rate * 100).toFixed(0)}%
            </p>
          </div>
        ))}
      </section>
    </div>
  );
}
```

### 5.3 Feature 2: Wire View File Watcher

**Current Issue:** The `/wire` page doesn't auto-refresh with new packets.

**Required Backend Endpoint:**
```python
# Add to src/api/server.py
from fastapi import FastAPI
from pathlib import Path
import json

@app.get("/api/wire/packets")
async def get_wire_packets():
    """List all packets in shared_channel directory."""
    shared_dir = Path("shared_channel")
    packets = []
    
    for f in sorted(shared_dir.glob("*.json")):
        if f.name.startswith("_"):  # Skip manifest
            continue
        with open(f) as fp:
            data = json.load(fp)
            data["filename"] = f.name
            packets.append(data)
    
    return {"packets": packets, "count": len(packets)}
```

**Frontend Changes:**
```typescript
// frontend/src/app/wire/page.tsx
// Add polling for /api/wire/packets
useEffect(() => {
  const interval = setInterval(async () => {
    const res = await fetch('http://localhost:8000/api/wire/packets');
    const data = await res.json();
    setPackets(data.packets);
  }, 2000);  // Poll every 2 seconds

  return () => clearInterval(interval);
}, []);
```

---

## 6. Section E: Testing & Quality Assurance

### 6.1 Current Test Coverage

| Component | Test File | Coverage |
|-----------|-----------|----------|
| Encoder | `tests/test_encoder.py` | Partial |
| Decoder | `tests/test_decoder.py` | Partial |
| GAN | None | 0% |
| RL | None | 0% |
| API | None | 0% |

### 6.2 Required Tests

#### Unit Tests Needed

```python
# tests/test_gan_generator.py
import pytest
import torch
from src.stealth.gan.generator import TemporalPatternGenerator, TimingSchedule

class TestTemporalPatternGenerator:
    def test_initialization(self):
        gen = TemporalPatternGenerator(num_channels=3)
        assert gen.num_channels == 3
        assert gen.latent_dim == 128
    
    def test_forward_output_shapes(self):
        gen = TemporalPatternGenerator(num_channels=3)
        z = torch.randn(4, 128)
        time = torch.randint(0, 24, (4,)).float()
        
        schedule = gen(z, sequence_length=20, time_of_day=time)
        
        assert schedule.delays.shape == (4, 20)
        assert schedule.channel_logits.shape == (4, 20, 3)
        assert schedule.confidence.shape == (4,)
    
    def test_delays_positive(self):
        gen = TemporalPatternGenerator(num_channels=3)
        schedule = gen.generate(batch_size=10, sequence_length=20)
        
        assert (schedule.delays >= 0).all()
    
    def test_channel_sampling(self):
        gen = TemporalPatternGenerator(num_channels=3)
        schedule = gen.generate(batch_size=10, sequence_length=20)
        channels = schedule.sample_channels()
        
        assert channels.min() >= 0
        assert channels.max() < 3


# tests/test_rl_environment.py
import pytest
import numpy as np
from src.stealth.rl.environment import StealthEnvironment

class TestStealthEnvironment:
    def test_reset(self):
        env = StealthEnvironment(num_channels=3)
        media = [f"media_{i}" for i in range(10)]
        
        state = env.reset(media)
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (env.state_dim,)
    
    def test_step_valid_action(self):
        env = StealthEnvironment(num_channels=3)
        env.reset([f"media_{i}" for i in range(10)])
        
        action = {"delay": 5.0, "channel": 0}
        next_state, reward, done, info = env.step(action)
        
        assert isinstance(reward, float)
        assert isinstance(done, bool)
    
    def test_episode_completion(self):
        env = StealthEnvironment(num_channels=3)
        env.reset([f"media_{i}" for i in range(3)])
        
        for _ in range(3):
            action = {"delay": 1.0, "channel": 0}
            _, _, done, _ = env.step(action)
        
        assert done  # Should complete after 3 items
```

#### Integration Tests Needed

```python
# tests/test_integration.py
import pytest
from pathlib import Path

class TestEndToEndEncoding:
    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding produce semantically similar results."""
        from src.engine.encoder import SemanticEncoder
        from src.engine.decoder import SemanticDecoder
        
        encoder = SemanticEncoder()
        decoder = SemanticDecoder()
        
        # Requires indices to be built
        encoder.load()
        decoder.load()
        
        message = "Meet at noon"
        encoded = encoder.encode(message, num_results=3)
        media_ids = [item.media_id for item in encoded.items]
        
        decoded = decoder.decode(media_ids)
        
        assert decoded.verification_rate > 0.8
        assert len(decoded.contents) == len(media_ids)
```

### 6.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_gan_generator.py -v
```

---

## 7. Section F: Production Readiness

### 7.1 Docker Optimization (See separate section below)

### 7.2 API Authentication (Low Priority)

**Current State:** No authentication
**Recommendation:** Add API key validation

```python
# src/api/server.py
from fastapi import Depends, HTTPException, Header

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("DCASS_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/api/encode", dependencies=[Depends(verify_api_key)])
async def encode_message(...):
    ...
```

### 7.3 Monitoring (Low Priority)

**Recommendation:** Add Prometheus metrics

```python
# src/api/metrics.py
from prometheus_client import Counter, Histogram

encode_requests = Counter('dcass_encode_requests_total', 'Total encode requests')
decode_requests = Counter('dcass_decode_requests_total', 'Total decode requests')
encode_latency = Histogram('dcass_encode_latency_seconds', 'Encode latency')
```

---

## 8. Appendix: File References

### Key Files for Training

| Purpose | File |
|---------|------|
| GAN Generator | `src/stealth/gan/generator.py` |
| GAN Trainer | `src/stealth/gan/trainer.py` |
| Warden (Discriminator) | `src/analysis/adversarial/warden.py` |
| RL Agent | `src/stealth/rl/agent.py` |
| RL Environment | `src/stealth/rl/environment.py` |
| Train GAN Script | `scripts/training/train_gan.py` |
| Train RL Script | `scripts/training/train_rl.py` |
| Generate Traffic | `scripts/training/generate_traffic_data.py` |

### Key Files for Decoding

| Purpose | File |
|---------|------|
| Semantic Decoder | `src/engine/decoder.py` |
| Unified Index | `src/corpus/index/unified_index.py` |
| Receiver Daemon | `scripts/runtime/run_receiver.py` |

### Docker Files

| Purpose | File |
|---------|------|
| Main Dockerfile | `Dockerfile` |
| Compose Config | `docker-compose.yml` |

---

## Quick Start Checklist

```
[ ] 1. Generate traffic data
    docker compose --profile training run dcass-gen-traffic

[ ] 2. Train GAN (50 epochs minimum)
    docker compose --profile training run dcass-train-gan

[ ] 3. Train RL (1000 episodes minimum)
    docker compose --profile training run dcass-train-rl

[ ] 4. Verify models exist
    ls models/gan/final.pt models/rl/ppo_agent_final.pt

[ ] 5. Update run_receiver.py with decoding integration

[ ] 6. Create frontend /decode page

[ ] 7. Run end-to-end test
    docker compose up
```

---

*Document generated: April 6, 2026*
