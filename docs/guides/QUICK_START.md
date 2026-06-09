# DCASS Quick Start Guide
## Running the Complete System (Backend + Frontend)

This guide shows you how to run the DCASS system **without Docker** for local development.

---

## 📋 Prerequisites

✅ Python 3.10+ installed  
✅ Node.js 18+ and npm installed  
✅ All Python dependencies installed (`pip install -r requirements.txt`)  
✅ Corpus indices built (or ready to build)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Start the Backend Server

Open a terminal and run:

```bash
# From the dcass/ directory
python scripts/runtime/start_server.py --reload
```

You should see:
```
======================================================================
🚀 Starting DCASS FastAPI Backend Server
======================================================================
📍 URL: http://127.0.0.1:8000
📖 API Docs: http://127.0.0.1:8000/docs
🔄 Auto-reload: Enabled
======================================================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Leave this terminal running!** The backend server must stay active.

---

### Step 2: Start the Frontend

Open a **new terminal** and run:

```bash
# Navigate to frontend directory
cd frontend

# Start Next.js development server
npm run dev
```

You should see:
```
   ▲ Next.js 14.2.0
   - Local:        http://localhost:3000
   - Ready in 2.1s
```

---

### Step 3: Open Your Browser

Visit: **http://localhost:3000**

You should see the DCASS home page with three navigation cards:
- 📊 **System Status**
- 🔐 **Encode Message**
- 📡 **Wire View**

---

## 🎯 Testing the Dynamic→Static Fallback

### Verify Static Mode (Default)

1. Go to **Status Dashboard**: http://localhost:3000/status

2. Check the "Stealth Models" card:
   - GAN Scheduler: 🔴 Not Trained
   - RL Agent: 🔴 Not Trained
   - Static Fallback: 🟢 Always Available

3. Check the "Stealth Mode" stat card:
   - Should show: **Static**

This confirms the system is using the **static fallback mode** (NoiseController).

---

### Test Encoding

1. Go to **Encode Interface**: http://localhost:3000/encode

2. Enter a message:
   ```
   people walking in a park on a sunny day
   ```

3. Select settings:
   - **Diversity Mode:** Best Match
   - **Modalities:** Image, Text (both selected)

4. Click **"🔐 Encode & Generate Sequence"**

5. You should see:
   - ✅ Encoding time (e.g., 150.5 ms)
   - ✅ Media items generated (e.g., 5 items)
   - ✅ Semantic chunks displayed
   - ✅ Modality breakdown (image/text distribution)
   - ✅ Full media sequence with IDs and scores

---

## 📡 Wire View (Packet Monitoring)

The Wire View shows real-time transmission telemetry. To see it in action:

### Option 1: Manual Sender Script

Open a **third terminal** and run:

```bash
python scripts/runtime/run_sender.py --mode auto --sequence-length 10
```

This will:
- Generate 10 media packets
- Write JSON files to `shared_channel/`
- Use auto mode (tries RL → GAN → Static)

### Option 2: Monitor Existing Packets

If `shared_channel/` already has packet files, you can view them by:

1. Going to: http://localhost:3000/wire
2. Clicking **"▶️ Start Monitoring"**

*(Note: Full real-time file watching requires a backend endpoint - coming soon)*

---

## 🔧 Troubleshooting

### "Connection Error" on Status Page

**Problem:** Frontend can't reach backend

**Solution:**
```bash
# Check if backend is running
curl http://localhost:8000/api/health

# Should return: {"status":"ok"}
```

If not running, restart with:
```bash
python scripts/runtime/start_server.py --reload
```

---

### "Index not built" in Status

**Problem:** Corpus indices are missing

**Solution:**
```bash
# Build indices (requires downloaded datasets)
python scripts/data/build_indices.py --modalities image text

# Or download sample dataset first
python scripts/data/download_flickr8k.py
```

---

### Port Already in Use

**Problem:** Port 8000 or 3000 is occupied

**Solution:**

For backend (port 8000):
```bash
python scripts/runtime/start_server.py --port 8001 --reload
```

For frontend (port 3000):
```bash
# In frontend/package.json, modify:
"dev": "next dev -p 3001"
```

---

## 📊 Understanding the Dynamic→Static Fallback

### How It Works

The system tries three scheduling modes in order:

```
Auto Mode Flow:
1. Try RL Agent (models/rl/ppo_agent_final.pt)
   ↓ (if checkpoint missing)
2. Fall back to GAN (models/gan/final.pt)
   ↓ (if checkpoint missing)
3. Fall back to Static (NoiseController) ← ALWAYS WORKS
```

### Current State

Without trained models:
- ✅ System uses **Static mode**
- ✅ NoiseController with behavioral profiles
- ✅ Mathematical jitter injection
- ✅ **Zero failure rate**

### To Enable Dynamic Mode

1. Generate training data:
   ```bash
   python scripts/training/generate_traffic_data.py --num-sessions 2000
   ```

2. Train GAN:
   ```bash
   python scripts/training/train_gan.py --epochs 50
   ```

3. Train RL:
   ```bash
   python scripts/training/train_rl.py --episodes 1000
   ```

4. Restart backend — system auto-detects checkpoints!

---

## 🎨 Frontend Features

### Home Page (`/`)
- Feature overview
- Navigation cards
- Key capabilities list

### Status Dashboard (`/status`)
- System health indicator
- Corpus statistics (total items, indices loaded)
- Device info (CPU/GPU)
- Stealth model status (GAN/RL/Static)
- Auto-refresh every 10 seconds

### Encode Interface (`/encode`)
- Message input textarea
- Diversity mode selector (Best/Round Robin/Balanced)
- Modality toggles (Image/Text/Audio)
- Real-time encoding results
- Semantic chunk display
- Media sequence with scores
- Modality breakdown visualization

### Wire View (`/wire`)
- Live transmission feed
- Packet statistics (total, avg delay, channels)
- Mode indicator badge
- Packet details (media_id, channel, sequence, delay, timestamp)
- Start/stop monitoring
- Clear packets button

---

## 🔐 API Endpoints Reference

Base URL: `http://localhost:8000/api`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/encode` | POST | Encode message → media sequence |
| `/decode` | POST | Decode media sequence → message |
| `/search` | POST | Search corpus by query |
| `/status` | GET | System status (indices, models, device) |
| `/benchmark/latest` | GET | Latest benchmark results |

### Example: Encode Request

```bash
curl -X POST http://localhost:8000/api/encode \
  -H "Content-Type: application/json" \
  -d '{
    "message": "secret message",
    "mode": "best",
    "modalities": ["image", "text"]
  }'
```

---

## 🎯 Next Steps

After getting the system running:

1. ✅ **Test encoding** with different messages
2. ✅ **Check status dashboard** for system health
3. ✅ **View wire telemetry** (when packets are generated)
4. ⏭️ **Build decode interface** (Bob's dashboard)
5. ⏭️ **Add real-time file watching** for Wire View
6. ⏭️ **Train AI models** (optional - for dynamic mode)

---

## 📚 Additional Resources

- **Main README:** `../README.md`
- **Frontend README:** `frontend/README.md`
- **Project Status:** `../document/PROJECT_COMPLETION_STATUS.md`
- **UIUX Requirements:** `../document/UIUX_BASE_REQUIREMENTS.md`
- **API Documentation:** http://localhost:8000/docs (when backend is running)

---

## ✅ Verification Checklist

- [ ] Backend server running on port 8000
- [ ] Frontend server running on port 3000
- [ ] Status page shows system health
- [ ] Corpus indices loaded (or at least acknowledged as missing)
- [ ] Stealth mode shows "Static" (with 🟢 green indicator)
- [ ] Encoding produces media sequences
- [ ] No console errors in browser DevTools
- [ ] API endpoints respond correctly

---

**Happy Testing!** 🚀

If you encounter any issues, check:
1. Both servers are running
2. No port conflicts
3. Dependencies installed (Python + npm)
4. Check browser console for errors

For detailed troubleshooting, see `PROJECT_COMPLETION_STATUS.md`.
