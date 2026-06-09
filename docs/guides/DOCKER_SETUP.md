# DCASS Docker Simulation Sandbox

This document explains how to use the Dockerized simulation environment for training and testing the DCASS steganography system.

## Overview

The Docker setup creates a simulated Alice-Bob network for training the RL agent:

- **Alice (Sender)**: Uses the RL agent to schedule transmissions
- **Bob (Receiver)**: Monitors shared channel and reassembles sequences
- **Shared Channel**: A volume that simulates the transmission wire

## Quick Start

### 1. Build the Docker Images

```bash
docker-compose build
```

### 2. Run the Simulation

**Training Mode** (Train the RL agent):
```bash
docker-compose run dcass-sender python scripts/runtime/run_sender.py --mode train --episodes 1000
```

**Transmission Mode** (Send a sequence using trained agent):
```bash
docker-compose up
```

This starts both Alice and Bob:
- Bob listens on the shared channel
- Alice sends media sequences using the RL agent

### 3. Monitor with TensorBoard (Optional)

```bash
docker-compose --profile monitoring up tensorboard
```

Then open http://localhost:6006 in your browser.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Alice (Sender) │         │  Bob (Receiver) │
│                 │         │                 │
│  RL Agent       │         │  Reassembly     │
│  └─> Scheduler  │         │  Buffer         │
│       └─> Send  │────────>│  └─> Decode     │
│                 │ Shared  │                 │
│                 │ Volume  │                 │
└─────────────────┘         └─────────────────┘
         │                           │
         └───────────┬───────────────┘
                     │
              ┌──────▼──────┐
              │   Warden    │
              │ (Evaluator) │
              └─────────────┘
```

## Components

### Dockerfile

Builds a Python 3.10 image with:
- PyTorch (CPU version for simulation)
- FAISS for semantic search
- Gymnasium for RL environment
- All DCASS dependencies

### docker-compose.yml

Defines three services:

1. **dcass-sender** (Alice):
   - Runs `scripts/runtime/run_sender.py`
   - Sends media sequences using RL scheduling
   - Writes packet metadata to `/app/storage/shared_channel`

2. **dcass-receiver** (Bob):
   - Runs `scripts/runtime/run_receiver.py`
   - Monitors `/app/storage/shared_channel` for incoming packets
   - Reassembles sequences after silence threshold

3. **tensorboard** (Optional):
   - Visualizes training metrics
   - Access at http://localhost:6006

### Shared Volume

The `shared_channel` directory simulates the network wire:
- Alice writes JSON packet metadata files
- Bob reads and processes them
- Asynchronous, out-of-order delivery is supported

## Training Workflow

### Phase 1: Train the Warden

```bash
docker-compose run dcass-sender python -c "
from src.stealth.gan.trainer import train_gan
from pathlib import Path

# Train Warden to detect bot traffic
train_gan(
    data_path=Path('data/human_traffic.json'),
    num_epochs=50
)
"
```

### Phase 2: Train the RL Agent

```bash
docker-compose run dcass-sender python scripts/runtime/run_sender.py \
    --mode train \
    --episodes 1000
```

This trains the RL agent to:
- Maximize throughput (items/minute)
- Minimize Warden detection score
- Respect channel rate limits

### Phase 3: Inference (Send Real Messages)

```bash
# Start receiver
docker-compose up dcass-receiver &

# Send a message
docker-compose run dcass-sender python scripts/runtime/run_sender.py \
    --mode rl \
    --message "Meet at the cafe at noon" \
    --checkpoint checkpoints/rl/trained_agent.pt
```

## Environment Variables

Set these in `docker-compose.yml` or pass via `-e`:

- `DCASS_ROLE`: `sender` or `receiver`
- `DCASS_SHARED_DIR`: Path to shared channel directory
- `CUDA_VISIBLE_DEVICES`: GPU device (use `""` for CPU)

## Monitoring

### Logs

Sender logs: `logs/sender/`
Receiver logs: `logs/receiver/`

### Checkpoints

Model checkpoints saved to `checkpoints/`:
- `checkpoints/gan/` - Warden and Generator
- `checkpoints/rl/` - RL agent policies

### Metrics

- Episode rewards
- Warden detection scores
- Throughput (items/minute)
- Transmission delays

## Troubleshooting

### Receiver not picking up packets

Check that both containers can access the shared volume:
```bash
docker-compose exec dcass-receiver ls -la /app/storage/shared_channel
docker-compose exec dcass-sender ls -la /app/storage/shared_channel
```

### Training is slow

The simulation uses CPU-only PyTorch. For faster training:
1. Build with GPU support in Dockerfile
2. Update `docker-compose.yml` to expose GPUs
3. Set `CUDA_VISIBLE_DEVICES=0`

### Out of memory

Reduce batch size in training configs:
- `PPOConfig.batch_size` (default: 64)
- `TrainingConfig.batch_size` (default: 32)

## Advanced Usage

### Custom Warden Training Data

Provide your own human traffic dataset:

```json
[
  {
    "delays": [5.2, 3.1, 12.4, ...],
    "channels": [0, 1, 0, 2, ...],
    "time_of_day": 14
  },
  ...
]
```

Save as `data/human_traffic.json` and mount in docker-compose.

### Multi-Channel Testing

Modify `num_channels` in the sender script:

```bash
docker-compose run dcass-sender python scripts/runtime/run_sender.py \
    --mode train \
    --episodes 500
```

Then update the environment initialization to use more channels.

## Next Steps

1. Integrate with real semantic encoder (requires FAISS indices)
2. Add real API channels (Twitter, Reddit, SMTP)
3. Implement persistent reassembly buffer across restarts
4. Add real-time monitoring dashboard

## References

- PPO Algorithm: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- GAN Training: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- DCASS Paper: See `CLAUDE.md` for project overview
