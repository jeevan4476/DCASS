# DCASS Quick Start Guide

> **Get DCASS up and running in 10 minutes**  
> Choose your setup: Local Development, Docker Web UI, or Docker Sender/Receiver Simulation

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required for local development |
| Node.js | 20+ | Required for frontend |
| Docker | 24.0+ | Required for Docker setups |
| Git | any | — |

---

## 🚀 Quick Setup Options

### **Option 1: Local Development (Recommended for Testing)**

This is the **fastest way** to get the web UI running locally.

#### **Step 1: Clone & Install**

```bash
# Clone the repository
git clone <repo-url>
cd dcass

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

#### **Step 2: Download Data & Build Indices**

**IMPORTANT:** You need the FAISS indices to run the system!

**Option A: Download Pre-built Indices (Fastest)** ⚡

*If you have access to pre-built indices from your team:*

```bash
# Copy indices to the right location
mkdir -p storage/data/indices
# Place image.index, image_metadata.json, text.index, text_metadata.json, audio.index, audio_metadata.json here
```

**Option B: Build From Scratch** 🔨

*This takes ~30-60 minutes depending on your hardware:*

```bash
# Download Flickr30k dataset (required!)
python scripts/data/download_flickr30k.py

# Build image index (~20 min)
python scripts/data/build_indices.py --modality image

# Build text index (~10 min)
python scripts/data/build_indices.py --modality text

# Build audio index (~30 min, optional)
python scripts/data/build_indices.py --modality audio
```

#### **Step 3: Start Backend API**

```bash
# Terminal 1: Start backend server
python -m uvicorn src.api.server:app --reload --port 8000

# You should see:
# INFO: Uvicorn running on http://127.0.0.1:8000
# 🔥 Warming up DCASS engine...
# ✅ DCASS engine ready!
```

**Backend will be available at:** `http://localhost:8000`

Test it: `curl http://localhost:8000/api/health`

#### **Step 4: Start Frontend**

```bash
# Terminal 2: Install frontend dependencies
cd frontend
npm install

# Start development server
npm run dev

# You should see:
# ▲ Next.js 14.x.x
# - Local: http://localhost:3000
```

**Frontend will be available at:** `http://localhost:3000`

#### **Step 5: Test the System**

1. Open browser: `http://localhost:3000`
2. Go to **Encode** page: `http://localhost:3000/encode`
3. Enter a message: `"Meet at the cafe at noon"`
4. Click **"Encode & Generate Sequence"**
5. Click **"📡 Transmit on Wire View"**
6. Watch packets appear live on Wire View! 🎉

---

### **Option 2: Docker Web UI (For Production-like Testing)**

Run the complete web stack (API + Frontend) in Docker.

#### **Step 1: Prepare Data**

⚠️ **IMPORTANT:** You must have indices built first!

```bash
# Make sure these files exist:
ls storage/data/indices/

# Expected output:
# image.index
# image_metadata.json
# text.index
# text_metadata.json
# audio.index
# audio_metadata.json
```

If missing, follow **Option 1 Step 2** to build them locally first.

#### **Step 2: Build & Start Services**

```bash
# Build Docker images (first time only, ~10 min)
docker compose --profile web build

# Start API + Frontend
docker compose --profile web up

# Services will start:
# - dcass-api: http://localhost:8000
# - dcass-frontend: http://localhost:3000
```

#### **Step 3: Access Web UI**

Open browser: `http://localhost:3000`

#### **Step 4: Stop Services**

```bash
# Stop containers
docker compose --profile web down
```

---

### **Option 3: Docker Sender/Receiver Simulation**

Run the Alice-Bob steganographic communication simulation.

#### **Prerequisites**

- Pre-built indices in `storage/data/indices/`
- Optional: Trained GAN/RL models in `storage/models/`

#### **Step 1: Build Images**

```bash
docker compose build
```

#### **Step 2: Run Simulation**

```bash
# Default: sender + receiver
docker compose up

# You should see:
# dcass-alice  | DCASS Sender (Alice)
# dcass-bob    | DCASS Receiver Daemon (Bob)
# dcass-alice  | TX #1  seq=0  media=media_000  ch=0  delay=2.0s  [static]
# dcass-bob    | Received packet: media_000 (seq=0, channel=0)
```

#### **Step 3: Monitor Logs**

```bash
# Watch Alice (sender)
docker compose logs -f dcass-sender

# Watch Bob (receiver)
docker compose logs -f dcass-receiver
```

#### **Step 4: Stop Simulation**

```bash
docker compose down
```

---

### **Option 4: Train AI Models (Advanced)**

Train the GAN and RL stealth schedulers.

#### **Generate Training Data**

```bash
docker compose --profile training run dcass-gen-traffic
```

#### **Train GAN Scheduler**

```bash
docker compose --profile training run dcass-train-gan
```

#### **Train RL Agent**

```bash
docker compose --profile training run dcass-train-rl
```

---

## 🎯 Feature Overview

### **Web UI Features**

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Overview and navigation |
| Encode | `/encode` | Encode messages into media sequences |
| Decode | `/decode` | Decode media sequences back to messages |
| Search | `/search` | Search the semantic corpus |
| Wire View | `/wire` | **NEW!** Live packet transmission monitoring |
| Status | `/status` | System health and indices status |

### **Wire View (NEW!)** 📡

The Wire View shows **real-time packet transmission** with:
- Live packet feed (packets appear with actual delays)
- Transmission progress bar
- Statistics (total packets, avg delay, channels, mode)
- Multi-channel distribution visualization

**How it works:**
1. Encode a message on `/encode`
2. Click "📡 Transmit on Wire View"
3. Watch packets appear one-by-one with realistic timing
4. Each packet has different timestamps (unlike before!)

---

## 🧪 Testing the System

### **Test 1: Health Check**

```bash
# Check backend
curl http://localhost:8000/api/health

# Expected: {"status":"ok"}
```

### **Test 2: Encode a Message**

```bash
curl -X POST http://localhost:8000/api/encode \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello World", "modalities": ["text", "image"]}'
```

### **Test 3: Wire View Transmission**

1. Go to `http://localhost:3000/encode`
2. Enter: `"The secret meeting is at midnight"`
3. Click "Encode & Generate Sequence"
4. Click "📡 Transmit on Wire View"
5. You should see packets appearing with **different timestamps**!

---

## 📊 What Data Do I Need?

### **Minimum (for Web UI)**

- **Image index**: 82 MB
- **Image metadata**: 25 MB
- **Text index**: 82 MB
- **Text metadata**: 11 MB

**Total: ~200 MB** (without audio)

### **Full System**

- All of the above +
- **Audio index**: 28 MB
- **Audio metadata**: 4 MB

**Total: ~232 MB**

### **Where to Get Data?**

1. **Download from team** (fastest) - Ask your teammates for the `storage/` folder
2. **Build locally** - Follow Option 1 Step 2 above
3. **Download from release** - Check GitHub releases (if available)

---

## 🐛 Troubleshooting

### **"ModuleNotFoundError: No module named 'clip'"**

```bash
pip install git+https://github.com/openai/CLIP.git
```

### **"Index not found" errors**

You need to build the indices first! See **Option 1 Step 2**.

### **Frontend can't connect to backend**

Make sure backend is running on port 8000:
```bash
curl http://localhost:8000/api/health
```

### **Docker "no indices found"**

The Docker containers need access to your local indices:
```bash
# Check if indices exist
ls storage/data/indices/

# If missing, build them locally first (see Option 1 Step 2)
```

### **Wire View shows packets with same timestamp**

This was a bug that's now **fixed**! Make sure you have the latest code.

The `/api/transmit` endpoint now uses **background threads** to write packets with real delays.

---

## 📚 Next Steps

- Read the [Complete Documentation](./docs/guides/COMPLETE_IMPLEMENTATION_GUIDE.md)
- Check [Script Execution Guide](./docs/guides/SCRIPT_EXECUTION_GUIDE.md)
- See [Team Handoff Guide](./docs/guides/TEAM_HANDOFF_GUIDE.md)
- Review [Project Status](./docs/project/PROJECT_COMPLETION_STATUS.md)

---

## 🤝 Need Help?

- Check the `docs/` folder for detailed guides
- Review error messages carefully
- Make sure all prerequisites are installed
- Ensure indices are built before running Docker

---

## ⚡ Quick Command Reference

```bash
# Local Development
python -m uvicorn src.api.server:app --reload --port 8000  # Backend
cd frontend && npm run dev                                  # Frontend

# Docker Web UI
docker compose --profile web up                             # Start
docker compose --profile web down                           # Stop

# Docker Simulation
docker compose up                                           # Start sender/receiver
docker compose down                                         # Stop

# Build Indices
python scripts/data/build_indices.py --modality image       # Images
python scripts/data/build_indices.py --modality text        # Text
python scripts/data/build_indices.py --modality audio       # Audio
```

---

**Last Updated:** April 7, 2026  
**Version:** 0.2.0 (Wire View Update)
