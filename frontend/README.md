# DCASS Frontend

Next.js-based web interface for the DCASS (Dynamic Context-Aware Semantic Steganography) system.

## Features

- **Status Dashboard**: System health, corpus statistics, model status
- **Encode Interface**: Alice's dashboard for message encoding
- **Wire View**: Real-time transmission telemetry and packet monitoring
- **Dark Mode UI**: Premium, technical aesthetic with neon accents

## Prerequisites

- Node.js 18+ and npm
- Python backend running on `localhost:8000`

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start the Backend Server

In a separate terminal, from the project root:

```bash
# Make sure you're in the dcass/ directory
python scripts/start_server.py --reload
```

The backend will start on `http://localhost:8000`

### 3. Start the Frontend

```bash
npm run dev
```

The frontend will start on `http://localhost:3000`

## Available Pages

| Route | Description |
|-------|-------------|
| `/` | Home page with feature overview |
| `/status` | System status dashboard |
| `/encode` | Message encoding interface (Alice) |
| `/decode` | Message decoding interface |
| `/wire` | Real-time packet transmission view |

## Dynamic → Static Fallback

The system automatically handles model availability:

1. **Auto Mode** (Recommended): Tries RL → GAN → Static
2. **RL Mode**: Tries RL → Static fallback
3. **GAN Mode**: Tries GAN → Static fallback
4. **Static Mode**: Always available (NoiseController)

The UI will show which mode is currently active with color-coded badges:
- 🟢 Green: RL or GAN active
- 🟡 Yellow: Static fallback active
- 🔴 Red: Model not available

## Development

```bash
# Development server with hot reload
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

## API Integration

The frontend connects to the FastAPI backend via `/api` endpoints:

- `GET /api/health` - Health check
- `POST /api/encode` - Encode message
- `POST /api/decode` - Decode sequence
- `POST /api/search` - Search corpus
- `GET /api/status` - System status

## Technologies

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Axios** - HTTP client
- **Recharts** - Data visualization (future use)
- **Framer Motion** - Animations (future use)

## Troubleshooting

### Backend Connection Error

If you see "Connection Error" on the status page:

1. Verify the backend is running: `python scripts/start_server.py --reload`
2. Check the backend URL: `http://localhost:8000/api/health`
3. Ensure no firewall is blocking port 8000

### CORS Issues

The backend has CORS enabled for all origins. If you still encounter CORS errors:

1. Check `src/api/server.py` has `allow_origins=["*"]`
2. Restart the backend server

### Empty Corpus

If indices are missing:

1. Run `python scripts/build_indices.py` to build corpus indices
2. Refresh the status page

## Project Structure

```
frontend/
├── src/
│   ├── app/              # Next.js App Router pages
│   │   ├── page.tsx      # Home page
│   │   ├── status/       # Status dashboard
│   │   ├── encode/       # Encode interface
│   │   ├── decode/       # Decode interface
│   │   └── wire/         # Wire view
│   ├── components/       # Reusable React components
│   │   ├── Navigation.tsx
│   │   └── UI.tsx
│   └── lib/              # Utilities and API client
│       └── api.ts
├── public/               # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── next.config.mjs
```

## License

Part of the DCASS project.
