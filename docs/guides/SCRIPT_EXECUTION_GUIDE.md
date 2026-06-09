# DCASS - Script Execution Guide
## Complete Reference for Docker & Local Execution

**Version:** 1.0  
**Date:** April 6, 2026  
**Audience:** Development Team, DevOps, Researchers

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start](#2-quick-start)
3. [Docker Execution](#3-docker-execution)
4. [Local Execution](#4-local-execution)
5. [Training Scripts](#5-training-scripts)
6. [Demo Scripts](#6-demo-scripts)
7. [Data Preparation](#7-data-preparation)
8. [Troubleshooting](#8-troubleshooting)
9. [Performance Optimization](#9-performance-optimization)
10. [Complete Command Reference](#10-complete-command-reference)

---

## 1. Prerequisites

### For Docker Execution

```bash
# Required
- Docker Engine 20.10+
- docker-compose 2.0+
- 16 GB RAM (recommended)
- 20 GB free disk space

# Verify installation
docker --version              # Should show: Docker version 20.10+
docker compose version        # Should show: Docker Compose version 2.0+
```

### For Local Execution

```bash
# Required
- Python 3.10+
- pip 23.0+
- Git
- 16 GB RAM (recommended)
- 20 GB free disk space

# Verify installation
python --version              # Should show: Python 3.10+
pip --version                 # Should show: pip 23.0+
git --version                 # Should show: git 2.30+
```

### Environment Setup

```bash
# Clone repository
git clone <repository-url> dcass
cd dcass

# Create environment file
cat > .env << 'EOF'
# Scheduler Configuration
DCASS_MODE=auto
DCASS_PROFILE=casual
DCASS_BASE_DELAY=3.0
DCASS_SEQ_LENGTH=20
NUM_CHANNELS=3

# Training Configuration
TRAFFIC_SESSIONS=2000
GAN_EPOCHS=50
GAN_BATCH_SIZE=32
RL_EPISODES=1000

# Receiver Configuration
RECEIVER_TIMEOUT=10

# Monitoring
TENSORBOARD_PORT=6006
EOF
```

---

## 2. Quick Start

### Option A: Docker (Recommended for Production)

```bash
# 1. Build images (first time only, ~5-10 minutes)
docker compose build

# 2. Run Alice-Bob simulation
docker compose up

# 3. In another terminal, watch the logs
docker compose logs -f dcass-alice
docker compose logs -f dcass-bob

# 4. Stop simulation
docker compose down
```

### Option B: Local (Recommended for Development)

```bash
# 1. Install dependencies (~10 minutes)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download datasets (optional, ~1-2 hours)
python scripts/data/download_flickr8k.py
python scripts/data/download_wikipedia.py --num-articles 5000

# 3. Build indices (~20-30 minutes)
python scripts/data/build_indices.py

# 4. Run demo
python scripts/demos/demo_dcass.py
```

---

## 3. Docker Execution

### 3.1 Building Images

```bash
# Build all images (production target)
docker compose build

# Build specific service
docker compose build dcass-sender

# Build with no cache (force rebuild)
docker compose build --no-cache

# Build development target (with dev tools)
docker compose build --target development

# Check image sizes
docker images | grep dcass
```

**Expected Sizes:**
- `dcass:sender` - ~2.5 GB
- `dcass:receiver` - ~2.5 GB
- `dcass:training` - ~2.5 GB

### 3.2 Running Simulation (Alice & Bob)

#### Basic Simulation

```bash
# Start sender (Alice) and receiver (Bob)
docker compose up

# Start in detached mode (background)
docker compose up -d

# View logs
docker compose logs -f

# View logs for specific service
docker compose logs -f dcass-alice
docker compose logs -f dcass-bob
```

#### Custom Configuration

```bash
# Set environment variables
DCASS_MODE=static \
DCASS_PROFILE=professional \
DCASS_SEQ_LENGTH=30 \
docker compose up

# Or modify .env file and restart
nano .env
docker compose up
```

#### Stop Simulation

```bash
# Stop containers (keep volumes)
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v

# Force stop (if containers are unresponsive)
docker compose down --timeout 1
```

### 3.3 Training with Docker

#### Step 1: Generate Training Data

```bash
# Generate 2000 sessions of synthetic traffic
docker compose --profile training run dcass-gen-traffic

# Custom configuration
docker compose --profile training run dcass-gen-traffic \
  python -u scripts/training/generate_traffic_data.py \
    --num-sessions 5000 \
    --num-channels 3 \
    --output /app/storage/data/behavioral/human_traffic.json

# Verify output
ls -lh data/behavioral/human_traffic.json
```

**Expected Output:**
```
data/behavioral/human_traffic.json  (~2-5 MB for 2000 sessions)
```

#### Step 2: Train GAN

```bash
# Train with default settings (50 epochs)
docker compose --profile training run dcass-train-gan

# Custom training
GAN_EPOCHS=100 GAN_BATCH_SIZE=64 \
docker compose --profile training run dcass-train-gan

# With WGAN-GP (more stable training)
docker compose --profile training run dcass-train-gan \
  python -u scripts/training/train_gan.py \
    --data /app/storage/data/behavioral/human_traffic.json \
    --epochs 100 \
    --batch-size 32 \
    --wgan-gp

# Resume from checkpoint
docker compose --profile training run dcass-train-gan \
  python -u scripts/training/train_gan.py \
    --resume /app/storage/models/gan/epoch_025.pt
```

**Training Time:** ~2-4 hours for 50 epochs (CPU)

**Expected Output:**
```
models/gan/
  epoch_000.pt
  epoch_001.pt
  ...
  epoch_049.pt
  final.pt
```

#### Step 3: Train RL Agent

```bash
# Train with default settings (1000 episodes)
docker compose --profile training run dcass-train-rl

# Custom training
RL_EPISODES=2000 \
docker compose --profile training run dcass-train-rl

# With pre-trained Warden (recommended)
docker compose --profile training run dcass-train-rl \
  python -u scripts/training/train_rl.py \
    --warden-checkpoint /app/storage/models/gan/final.pt \
    --episodes 2000 \
    --lambda-stealth 100.0
```

**Training Time:** ~4-8 hours for 1000 episodes (CPU)

**Expected Output:**
```
models/rl/
  ppo_agent_final.pt
```

#### Step 4: Monitor Training (TensorBoard)

```bash
# Start TensorBoard
docker compose --profile monitoring up tensorboard

# Access TensorBoard
open http://localhost:6006

# Custom port
TENSORBOARD_PORT=6007 docker compose --profile monitoring up tensorboard
```

**Metrics to Monitor:**
- GAN: `generator_loss`, `warden_loss`, `fake_bot_prob`
- RL: `episode_reward`, `warden_score`, `policy_loss`

### 3.4 Complete Training Pipeline

```bash
# Run all training steps sequentially
# Step 1: Generate data
docker compose --profile training run dcass-gen-traffic

# Step 2: Train GAN (2-4 hours)
docker compose --profile training run dcass-train-gan

# Step 3: Train RL with trained Warden (4-8 hours)
docker compose --profile training run dcass-train-rl

# Step 4: Verify models
ls -lh models/gan/final.pt models/rl/ppo_agent_final.pt

# Step 5: Test with trained models
DCASS_MODE=rl docker compose up
```

### 3.5 Docker Utility Commands

```bash
# View running containers
docker compose ps

# Execute command in running container
docker compose exec dcass-alice bash

# View container resource usage
docker stats

# Clean up Docker system
docker system prune -a  # Remove all unused images
docker volume prune     # Remove unused volumes

# Export container logs
docker compose logs > dcass_logs.txt

# Inspect container
docker inspect dcass-alice
```

---

## 4. Local Execution

### 4.1 Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows (Command Prompt):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import faiss; print(faiss.__version__)"
```

**Installation Time:** ~10-15 minutes

### 4.2 Data Preparation

#### Download Datasets

```bash
# Flickr8k (required, ~1 GB, ~30 minutes)
python scripts/data/download_flickr8k.py

# Flickr30k (optional, ~4 GB, ~2 hours)
python scripts/data/download_flickr30k.py

# Wikipedia (required, ~500 MB, ~30 minutes)
python scripts/data/download_wikipedia.py --num-articles 5000

# Audio datasets (optional, ~2 GB, ~1 hour)
python scripts/audio/audio_step1_download.py
```

**Storage Requirements:**
- Flickr8k: ~1 GB
- Flickr30k: ~4 GB
- Wikipedia: ~500 MB
- Audio: ~2 GB
- **Total:** ~7.5 GB

#### Build Indices

```bash
# Build all indices (~20-30 minutes)
python scripts/data/build_indices.py

# Build specific indices
python scripts/build_flickr8k_index.py      # Image index
python scripts/data/add_wikipedia_to_index.py    # Text index
python scripts/audio/audio_step2_build_index.py   # Audio index

# Verify indices
ls -lh data/indices/
# Expected: image.index, image_metadata.json, text.index, text_metadata.json
```

**Expected Sizes:**
- `image.index`: ~50-200 MB
- `text.index`: ~20-50 MB
- `audio.index`: ~100-300 MB

### 4.3 Running Core Scripts

#### Demo Scripts

```bash
# Full demo (encoding + decoding + distribution)
python scripts/demos/demo_dcass.py

# Encoder demo only
python scripts/demos/demo_encoder.py

# Full loop demo (with visualization)
python scripts/demos/demo_full_loop.py

# Test encoding
python scripts/testing/test_encoding.py

# Test stealth system
python scripts/testing/test_stealth_system.py
```

#### API Server

```bash
# Start FastAPI server
python scripts/runtime/start_server.py

# Custom host/port
python scripts/runtime/start_server.py --host 0.0.0.0 --port 8080

# With auto-reload (development)
python scripts/runtime/start_server.py --reload

# Test endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/api/status
```

#### CLI Usage

```bash
# System status
python src/cli/main.py status

# Encode message
python src/cli/main.py encode "Secret message" \
  --num-results 5 \
  --diversity balanced

# Decode sequence
python src/cli/main.py decode \
  --media-ids flickr8k_00123 wiki_00456

# Search corpus
python src/cli/main.py search "sunset" \
  --modality image \
  --k 10

# Run demo
python src/cli/main.py demo \
  --message "Test message"
```

---

## 5. Training Scripts

### 5.1 Generate Training Data

```bash
# Default (2000 sessions)
python scripts/training/generate_traffic_data.py

# Custom configuration
python scripts/training/generate_traffic_data.py \
  --num-sessions 5000 \
  --num-channels 3 \
  --seed 42 \
  --output data/behavioral/human_traffic.json

# Verify output
wc -l data/behavioral/human_traffic.json
python -c "import json; data = json.load(open('data/behavioral/human_traffic.json')); print(f'{len(data)} sessions')"
```

**Output Format:**
```json
[
  {
    "delays": [5.2, 3.1, 12.4, ...],
    "channels": [0, 1, 0, 2, ...],
    "time_of_day": 14,
    "num_channels": 3,
    "session_length": 25
  },
  ...
]
```

### 5.2 Train GAN

```bash
# Basic training
python scripts/training/train_gan.py \
  --data data/behavioral/human_traffic.json \
  --epochs 50 \
  --batch-size 32 \
  --checkpoint-dir models/gan

# Advanced training (WGAN-GP)
python scripts/training/train_gan.py \
  --data data/behavioral/human_traffic.json \
  --epochs 100 \
  --batch-size 64 \
  --lr-gen 1e-4 \
  --lr-warden 2e-4 \
  --warden-steps 5 \
  --wgan-gp \
  --device cuda

# Resume from checkpoint
python scripts/training/train_gan.py \
  --resume models/gan/epoch_025.pt

# Monitor with TensorBoard
tensorboard --logdir logs/train-gan --port 6006
```

**Training Arguments:**
- `--data`: Path to traffic data JSON
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr-gen`: Generator learning rate (default: 1e-4)
- `--lr-warden`: Warden learning rate (default: 2e-4)
- `--warden-steps`: Warden updates per generator update (default: 5)
- `--wgan-gp`: Enable WGAN-GP training
- `--device`: Training device (cpu/cuda)
- `--checkpoint-dir`: Output directory
- `--resume`: Resume from checkpoint

**Expected Output:**
```
Loading traffic data from data/behavioral/human_traffic.json ...
  2000 sessions, 63 batches/epoch
Starting GAN training for 50 epochs on cpu
Generator params: 1,234,567
Warden params: 2,345,678

[Epoch 0][0/63] G_loss: 0.6543 | W_loss: 0.3210 | Real: 0.234 | Fake: 0.876 | Conf: 0.512
...
Checkpoint saved: models/gan/epoch_000.pt
...
Training complete. Checkpoints in models/gan
```

### 5.3 Train RL Agent

```bash
# Basic training
python scripts/training/train_rl.py \
  --episodes 1000 \
  --checkpoint-dir models/rl

# With pre-trained Warden (recommended)
python scripts/training/train_rl.py \
  --warden-checkpoint models/gan/final.pt \
  --episodes 2000 \
  --num-channels 3 \
  --lambda-stealth 100.0 \
  --device cuda

# Custom configuration
python scripts/training/train_rl.py \
  --episodes 2000 \
  --num-channels 3 \
  --lambda-stealth 50.0 \
  --lr 3e-4 \
  --log-interval 10 \
  --device cpu
```

**Training Arguments:**
- `--episodes`: Number of training episodes (default: 1000)
- `--num-channels`: Number of channels (default: 3)
- `--warden-checkpoint`: Path to trained Warden
- `--lambda-stealth`: Stealth penalty coefficient (default: 100.0)
- `--lr`: Learning rate (default: 3e-4)
- `--device`: Training device (cpu/cuda)
- `--checkpoint-dir`: Output directory
- `--log-interval`: Episodes between logging (default: 20)

**Expected Output:**
```
Loading trained Warden from models/gan/final.pt
Actor-Critic params: 345,678
Training PPO agent for 1000 episodes...

Episode 100/1000 | Avg Reward: 45.23 | Avg Length: 18.2 | Warden Score: 0.312 | Policy Loss: 0.0234
Episode 200/1000 | Avg Reward: 62.41 | Avg Length: 19.8 | Warden Score: 0.245 | Policy Loss: 0.0189
...
Training complete. Agent saved to models/rl/ppo_agent_final.pt
```

### 5.4 Evaluate Models

```bash
# Evaluate stealth system
python scripts/testing/evaluate_stealth.py \
  --gan-checkpoint models/gan/final.pt \
  --rl-checkpoint models/rl/ppo_agent_final.pt \
  --num-episodes 100

# Benchmark
python scripts/run_benchmark.py
```

---

## 6. Demo Scripts

### 6.1 Full System Demo

```bash
# Complete demo (encoding, distribution, decoding)
python scripts/demos/demo_dcass.py

# Custom message
python scripts/demos/demo_dcass.py \
  --message "Meet at the secret location" \
  --num-results 5

# With trained models
python scripts/demos/demo_dcass.py \
  --message "Secret message" \
  --mode rl \
  --rl-checkpoint models/rl/ppo_agent_final.pt
```

**Expected Output:**
```
════════════════════════════════════════════
DCASS Full System Demo
════════════════════════════════════════════

Message: "Meet at the secret location"

[Encoding]
✓ Chunked into 5 segments
✓ Encoded to 15 media items
  - 7 images
  - 5 text
  - 3 audio

[Distribution]
✓ Scheduled with mode: static
  Total delay: 87.5 seconds

[Decoding]
✓ Decoded 15 items
✓ Verification: 100%
✓ Reconstructed: "Meet at secret location"

════════════════════════════════════════════
```

### 6.2 Encoder Demo

```bash
# Test encoding only
python scripts/demos/demo_encoder.py

# Custom configuration
python scripts/demos/demo_encoder.py \
  --message "Test message" \
  --num-results 10 \
  --diversity round_robin
```

### 6.3 Full Loop Demo

```bash
# Complete encode-decode loop
python scripts/demos/demo_full_loop.py

# With visualization
python scripts/demos/demo_full_loop.py --visualize
```

---

## 7. Data Preparation

### 7.1 Dataset Download Workflow

```bash
# Step 1: Create data directory
mkdir -p data/raw data/indices data/behavioral

# Step 2: Download Flickr8k (required)
python scripts/data/download_flickr8k.py
# Output: data/raw/flickr8k/

# Step 3: Download Wikipedia (required)
python scripts/data/download_wikipedia.py --num-articles 5000
# Output: data/raw/wikipedia/

# Step 4: Download Flickr30k (optional)
python scripts/data/download_flickr30k.py
# Output: data/raw/flickr30k/

# Step 5: Download audio (optional)
python scripts/audio/audio_step1_download.py
# Output: data/raw/audio/

# Verify downloads
du -sh data/raw/*
```

### 7.2 Index Building Workflow

```bash
# Option 1: Build all indices at once
python scripts/data/build_indices.py

# Option 2: Build incrementally
# Build image index (Flickr8k)
python scripts/build_flickr8k_index.py

# Add Flickr30k to image index
python scripts/data/build_flickr30k_index.py

# Build text index (Wikipedia)
python scripts/data/add_wikipedia_to_index.py

# Build audio index
python scripts/audio/audio_step2_build_index.py

# Verify indices
ls -lh data/indices/
python -c "from src.corpus.index.unified_index import UnifiedSemanticIndex; idx = UnifiedSemanticIndex(); print(idx.load())"
```

**Index Building Time:**
- Flickr8k: ~5-10 minutes
- Flickr30k: ~20-30 minutes
- Wikipedia (5k articles): ~10-15 minutes
- Audio: ~15-20 minutes
- **Total:** ~50-75 minutes

### 7.3 Data Verification

```bash
# Check index status
python -c "
from src.corpus.index.unified_index import UnifiedSemanticIndex
idx = UnifiedSemanticIndex()
status = idx.load()
print('Index Status:', status)
"

# Test search
python -c "
from src.corpus.index.unified_index import UnifiedSemanticIndex
idx = UnifiedSemanticIndex()
idx.load()
results = idx.search('sunset', modality='image', k=5)
for r in results:
    print(f'{r.media_id}: {r.score:.3f}')
"

# Test encoding
python scripts/testing/test_encoding.py
```

---

## 8. Troubleshooting

### 8.1 Docker Issues

#### Issue: Docker build fails with "out of disk space"

```bash
# Solution: Clean up Docker system
docker system prune -a
docker volume prune

# Check disk usage
docker system df
```

#### Issue: Container exits immediately

```bash
# Check logs
docker compose logs dcass-alice

# Run interactively
docker compose run --rm dcass-alice bash

# Check entrypoint
docker inspect dcass-alice | grep -A 5 "Entrypoint"
```

#### Issue: "Cannot connect to Docker daemon"

```bash
# Start Docker daemon
sudo systemctl start docker  # Linux
open -a Docker               # macOS
```

### 8.2 Python/Dependency Issues

#### Issue: ImportError for torch/faiss

```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Reinstall FAISS
pip uninstall faiss-cpu
pip install faiss-cpu==1.7.4
```

#### Issue: CLIP installation fails

```bash
# Install CLIP from source
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git@main
```

#### Issue: "No module named 'src'"

```bash
# Set PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Or use python -m
python -m src.cli.main status
```

### 8.3 Training Issues

#### Issue: GAN training - generator loss goes to 0

```bash
# Solution: Mode collapse, use WGAN-GP
python scripts/training/train_gan.py --wgan-gp

# Or reduce warden learning rate
python scripts/training/train_gan.py --lr-warden 1e-4
```

#### Issue: RL training - reward not improving

```bash
# Solution: Reduce stealth penalty
python scripts/training/train_rl.py --lambda-stealth 50.0

# Or increase learning rate
python scripts/training/train_rl.py --lr 1e-3
```

#### Issue: Out of memory during training

```bash
# Reduce batch size
python scripts/training/train_gan.py --batch-size 16

# Or use CPU
python scripts/training/train_gan.py --device cpu
```

### 8.4 Index Issues

#### Issue: FAISS index not found

```bash
# Rebuild indices
python scripts/data/build_indices.py

# Check index directory
ls -la data/indices/
```

#### Issue: Empty search results

```bash
# Verify index loaded
python -c "
from src.corpus.index.unified_index import UnifiedSemanticIndex
idx = UnifiedSemanticIndex()
status = idx.load()
print('Loaded:', status)
print('Stats:', idx.stats())
"
```

---

## 9. Performance Optimization

### 9.1 Docker Performance

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker compose build

# Use layer caching
docker compose build --cache-from dcass:latest

# Limit resources
docker compose up --scale dcass-sender=1 \
  --memory=4g \
  --cpus=2
```

### 9.2 Training Performance

```bash
# Use GPU if available
python scripts/training/train_gan.py --device cuda

# Increase batch size (if enough memory)
python scripts/training/train_gan.py --batch-size 64

# Reduce logging frequency
python scripts/training/train_rl.py --log-interval 50

# Use multiple workers
python scripts/training/train_gan.py --num-workers 4
```

### 9.3 Index Performance

```bash
# Use FAISS GPU (if available)
pip install faiss-gpu

# Optimize index search
python -c "
from src.corpus.index.unified_index import UnifiedSemanticIndex
idx = UnifiedSemanticIndex()
idx.load()
# Use batch search for multiple queries
queries = ['sunset', 'beach', 'mountain']
for q in queries:
    results = idx.search(q, modality='image', k=10)
"
```

---

## 10. Complete Command Reference

### Docker Commands

```bash
# Build & Run
docker compose build                                  # Build all images
docker compose up                                     # Start simulation
docker compose up -d                                  # Start in background
docker compose down                                   # Stop simulation
docker compose down -v                                # Stop and remove volumes

# Training
docker compose --profile training run dcass-gen-traffic   # Generate data
docker compose --profile training run dcass-train-gan     # Train GAN
docker compose --profile training run dcass-train-rl      # Train RL

# Monitoring
docker compose --profile monitoring up tensorboard    # Start TensorBoard
docker compose logs -f                                # View logs
docker compose ps                                     # List containers
docker compose exec dcass-alice bash                  # Shell into container

# Cleanup
docker system prune -a                                # Remove all images
docker volume prune                                   # Remove volumes
```

### Python Scripts

```bash
# Data Preparation
python scripts/data/download_flickr8k.py
python scripts/data/download_flickr30k.py
python scripts/data/download_wikipedia.py --num-articles 5000
python scripts/audio/audio_step1_download.py
python scripts/data/build_indices.py

# Training
python scripts/training/generate_traffic_data.py --num-sessions 2000
python scripts/training/train_gan.py --epochs 50 --batch-size 32
python scripts/training/train_rl.py --episodes 1000

# Demo
python scripts/demos/demo_dcass.py
python scripts/demos/demo_encoder.py
python scripts/demos/demo_full_loop.py

# Testing
python scripts/testing/test_encoding.py
python scripts/testing/test_stealth_system.py

# Server
python scripts/runtime/start_server.py --host 0.0.0.0 --port 8000
```

### CLI Commands

```bash
# System
python src/cli/main.py status
python src/cli/main.py benchmark

# Encoding/Decoding
python src/cli/main.py encode "message" --num-results 5
python src/cli/main.py decode --media-ids m1 m2 m3
python src/cli/main.py demo --message "test"

# Search
python src/cli/main.py search "query" --modality image --k 10
python src/cli/main.py verify --media-ids m1 m2 m3

# Distribution
python src/cli/main.py distribute --media-ids m1 m2 m3 --channels console
```

---

## Appendix A: Environment Variables Reference

```bash
# Scheduler
DCASS_MODE=auto|rl|gan|static
DCASS_PROFILE=casual|professional|gamer|night_owl
DCASS_BASE_DELAY=3.0
DCASS_SEQ_LENGTH=20
NUM_CHANNELS=3

# Training
TRAFFIC_SESSIONS=2000
GAN_EPOCHS=50
GAN_BATCH_SIZE=32
RL_EPISODES=1000

# Paths
DCASS_SHARED_DIR=./shared_channel
PYTHONPATH=/app

# Performance
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=""
```

## Appendix B: File Paths Reference

```bash
# Data
data/raw/flickr8k/           # Flickr8k dataset
data/raw/wikipedia/          # Wikipedia articles
data/indices/                # FAISS indices
data/behavioral/             # Training data

# Models
models/gan/final.pt          # Trained GAN
models/rl/ppo_agent_final.pt # Trained RL agent

# Logs
logs/sender/                 # Alice logs
logs/receiver/               # Bob logs
logs/train-gan/              # GAN training logs
logs/train-rl/               # RL training logs

# Shared
shared_channel/              # Alice-Bob communication
```

---

*Script Execution Guide - Version 1.0*  
*Last Updated: April 6, 2026*  
*For issues or questions, see troubleshooting section or contact the development team*
