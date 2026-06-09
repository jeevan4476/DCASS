# Running Frontend & Backend - Quick Start Guide

**Date:** April 6, 2026  
**Purpose:** Complete guide to run DCASS frontend and backend locally

---

## Quick Start (TL;DR)

### Terminal 1 - Backend (FastAPI)
```bash
# From project root
python scripts/runtime/start_server.py --reload
```
**Backend will run at:** http://localhost:8000

### Terminal 2 - Frontend (Next.js)
```bash
# From project root
cd frontend
npm run dev
```
**Frontend will run at:** http://localhost:3000

---

## Detailed Instructions

## Part 1: Backend (FastAPI)

### Prerequisites

```bash
# Check Python version (need 3.10+)
python --version

# Make sure you're in project root
cd C:\Users\kappa\OneDrive\capstone\dcass

# Activate virtual environment (if using one)
# If using venv:
.\vir\Scripts\activate

# OR if using conda:
conda activate dcass
```

### Install Backend Dependencies

```bash
# Install all Python dependencies
pip install -r requirements.txt

# Verify uvicorn is installed (for FastAPI)
pip list | grep uvicorn
```

### Start Backend Server

**Option 1: Using the script (Recommended)**
```bash
python scripts/runtime/start_server.py --reload
```

**Option 2: Using uvicorn directly**
```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```

**Option 3: Custom host/port**
```bash
# Allow external connections
python scripts/runtime/start_server.py --host 0.0.0.0 --port 8000 --reload

# Different port
python scripts/runtime/start_server.py --port 8080 --reload
```

### Backend Endpoints

Once running, access:

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Root endpoint |
| http://localhost:8000/docs | Interactive API docs (Swagger) |
| http://localhost:8000/redoc | Alternative API docs |
| http://localhost:8000/health | Health check |
| http://localhost:8000/api/status | System status |

### Test Backend

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","version":"0.1.0"}
```

---

## Part 2: Frontend (Next.js)

### Prerequisites

```bash
# Check Node.js version (need 18+)
node --version

# Check npm
npm --version

# Navigate to frontend directory
cd frontend
```

### Install Frontend Dependencies

**First time only:**
```bash
cd frontend

# Install dependencies (takes 1-2 minutes)
npm install

# Or if using yarn
yarn install
```

### Start Frontend Development Server

```bash
# From frontend directory
npm run dev

# Or using yarn
yarn dev
```

**Frontend will start at:** http://localhost:3000

### Frontend Pages

Once running, access:

| URL | Page | Status |
|-----|------|--------|
| http://localhost:3000 | Dashboard / Status | ✅ Working |
| http://localhost:3000/encode | Encode Message | ✅ Working |
| http://localhost:3000/wire | Wire View (Traffic) | ✅ Working |
| http://localhost:3000/decode | Decode Message | ✅ Working |

### Build for Production

```bash
cd frontend

# Create production build
npm run build

# Start production server
npm start
```

---

## Part 3: Running Both Together

### Method 1: Two Terminals (Recommended for Development)

**Terminal 1 - Backend:**
```powershell
# Windows PowerShell
cd C:\Users\kappa\OneDrive\capstone\dcass
.\vir\Scripts\activate
python scripts/runtime/start_server.py --reload
```

**Terminal 2 - Frontend:**
```powershell
# Windows PowerShell
cd C:\Users\kappa\OneDrive\capstone\dcass\frontend
npm run dev
```

### Method 2: Using Docker Compose

```bash
# From project root
docker compose up

# Backend: http://localhost:8000
# Frontend: You'd need to add frontend to docker-compose.yml
```

### Method 3: Background Processes (Windows)

```powershell
# Start backend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\kappa\OneDrive\capstone\dcass; .\vir\Scripts\activate; python scripts/runtime/start_server.py --reload"

# Start frontend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\kappa\OneDrive\capstone\dcass\frontend; npm run dev"
```

---

## Configuration

### Backend Configuration

Edit `config/default.yaml` to change:
- Ports
- Data paths
- Model settings
- Logging level

### Frontend Configuration

Edit `frontend/src/lib/api.ts` to change backend URL:

```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
```

Or set environment variable:
```bash
# Create frontend/.env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > frontend/.env.local
```

---

## Troubleshooting

### Backend Issues

#### Port Already in Use
```bash
# Error: Address already in use
# Solution: Use different port
python scripts/runtime/start_server.py --port 8001
```

#### Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Install dependencies
pip install -r requirements.txt

# Or check Python path
echo $PYTHONPATH
```

#### Virtual Environment Issues
```bash
# Deactivate and reactivate
deactivate
.\vir\Scripts\activate

# Or recreate venv
python -m venv vir
.\vir\Scripts\activate
pip install -r requirements.txt
```

### Frontend Issues

#### Port 3000 Already in Use
```bash
# Error: Port 3000 is already in use
# Solution: Use different port
PORT=3001 npm run dev

# Or kill process using port 3000
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

#### Module Not Found
```bash
# Error: Cannot find module
# Solution: Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

#### TypeScript Errors
```bash
# Check TypeScript config
cat tsconfig.json

# Rebuild
npm run build
```

#### Connection to Backend Failed
```bash
# Check backend is running
curl http://localhost:8000/health

# Check frontend API URL
cat frontend/src/lib/api.ts

# Check CORS settings in backend
# Should allow localhost:3000
```

---

## Development Workflow

### Typical Development Session

1. **Start Backend:**
   ```bash
   python scripts/runtime/start_server.py --reload
   ```

2. **Start Frontend:**
   ```bash
   cd frontend && npm run dev
   ```

3. **Open Browser:**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

4. **Make Changes:**
   - Backend changes auto-reload (with `--reload` flag)
   - Frontend changes auto-refresh (Next.js hot reload)

5. **Test:**
   - Use frontend UI
   - Or use API docs (http://localhost:8000/docs)
   - Or use curl/Postman

### Testing the Full Stack

**Test 1: Backend Health Check**
```bash
curl http://localhost:8000/health
```

**Test 2: Frontend Connects to Backend**
1. Open http://localhost:3000
2. Navigate to Status page
3. Should show system status from backend

**Test 3: Encode Flow**
1. Go to http://localhost:3000/encode
2. Enter a message
3. Click "Encode"
4. Backend should process and return sequence

---

## API Endpoints (Backend)

### Available Endpoints

```bash
# Health check
GET http://localhost:8000/health

# System status
GET http://localhost:8000/api/status

# Encode message
POST http://localhost:8000/api/encode
Body: {"message": "Hello World", "modality": "image"}

# Decode sequence
POST http://localhost:8000/api/decode
Body: {"sequence": ["id1", "id2", "id3"], "modality": "image"}

# Search corpus
POST http://localhost:8000/api/search
Body: {"query": "dog playing", "modality": "image", "k": 5}

# List available channels
GET http://localhost:8000/api/channels

# Get sender status
GET http://localhost:8000/api/sender/status

# Get receiver status
GET http://localhost:8000/api/receiver/status
```

### Test with curl

```bash
# Encode a message
curl -X POST http://localhost:8000/api/encode \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello World", "modality": "image"}'

# Get status
curl http://localhost:8000/api/status
```

---

## Environment Variables

### Backend (.env)

Create `.env` in project root:
```bash
# Server
HOST=127.0.0.1
PORT=8000

# Data paths
DATA_DIR=storage/data
MODELS_DIR=storage/models

# Logging
LOG_LEVEL=INFO
```

### Frontend (.env.local)

Create `frontend/.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Production Deployment

### Backend Production

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with gunicorn (production WSGI server)
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.server:app --bind 0.0.0.0:8000
```

### Frontend Production

```bash
cd frontend

# Build production bundle
npm run build

# Start production server
npm start

# Or use pm2 (process manager)
npm install -g pm2
pm2 start "npm start" --name dcass-frontend
```

---

## Quick Reference

### Commands Summary

| Action | Command |
|--------|---------|
| **Backend** | |
| Start dev server | `python scripts/runtime/start_server.py --reload` |
| Start production | `uvicorn src.api.server:app --host 0.0.0.0 --port 8000` |
| Test backend | `curl http://localhost:8000/health` |
| **Frontend** | |
| Install deps | `cd frontend && npm install` |
| Start dev server | `npm run dev` |
| Build production | `npm run build` |
| Start production | `npm start` |
| **Both** | |
| Stop all | `Ctrl+C` in both terminals |

### URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Status Page | http://localhost:3000 |
| Encode Page | http://localhost:3000/encode |
| Wire View | http://localhost:3000/wire |

---

## Next Steps

After getting both running:

1. **Test the System:**
   - Go to http://localhost:3000/encode
   - Try encoding a message
   - Check wire view for traffic

2. **Download Data (if needed):**
   ```bash
   make download-data
   make build-index
   ```

3. **Train Models (optional):**
   ```bash
   python scripts/training/generate_traffic_data.py
   python scripts/training/train_gan.py
   python scripts/training/train_rl.py
   ```

4. **Run Full Demo:**
   ```bash
   python scripts/demos/demo_dcass.py "Your message here"
   ```

---

**You're all set! Both frontend and backend should now be running!** 🚀

**Access the application at:** http://localhost:3000
