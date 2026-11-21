# DCASS Frontend

Next.js 15 web interface for the **Distributed Covert Audio Steganography System (DCASS)** — a research platform for DSSC/VCP covert audio encoding, RS-ECC erasure recovery, and real-time chat.

---

## Tech Stack

| Layer | Library / Version |
|---|---|
| Framework | Next.js 15.5.x (App Router, standalone output) |
| UI | React 18, Tailwind CSS, Framer Motion |
| Charts | Recharts |
| Icons | Lucide React |
| HTTP | Axios 1.21+ |
| Runtime | Node.js 20 / Bun 1 |

---

## Prerequisites

- **Node.js ≥ 20** or **Bun ≥ 1.0**
- Backend API running at `http://localhost:8000` (see [`docs/guides/GETTING_STARTED.md`](../docs/guides/GETTING_STARTED.md))

---

## Getting Started

### Install dependencies

```bash
bun install          # preferred
# or
npm install
```

### Development server

```bash
bun run dev
# or
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Production build

```bash
bun run build
bun run start
```

---

## Environment Variables

Create a `.env.local` file in this directory:

```env
# Public API base URL (used by the browser)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Internal API URL used by Next.js rewrites inside Docker Compose
# (overrides NEXT_PUBLIC_API_URL for server-side proxying)
API_INTERNAL_URL=http://dcass-api:8000
```

The dev server proxies all `/api/*` requests to the backend via `next.config.mjs` rewrites, so no CORS configuration is needed.

---

## Docker

The frontend ships with a multi-stage Dockerfile:

```bash
# Build and run via Docker Compose (recommended)
docker compose --profile web up dcass-frontend

# Or build the image directly
docker build -t dcass-frontend ./frontend
docker run -p 3001:3001 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  dcass-frontend
```

See [`docs/guides/DOCKER_SETUP.md`](../docs/guides/DOCKER_SETUP.md) for the full stack.

---

## Pages

| Route | Description |
|---|---|
| `/` | Dashboard / status overview |
| `/encode` | DSSC covert audio encode |
| `/decode` | DSSC covert audio decode |
| `/erasure` | RS-ECC erasure encode & recover |
| `/chat` | Real-time covert chat |
| `/wire` | Wire protocol inspector |
| `/inbox` | Message inbox |
| `/status` | System health & diagnostics |
| `/login` | Authentication |
| `/register` | User registration |

---

## Project Structure

```
frontend/
├── src/
│   ├── app/            # Next.js App Router pages
│   │   ├── encode/     # DSSC encode UI
│   │   ├── decode/     # DSSC decode UI
│   │   ├── erasure/    # RS-ECC erasure UI
│   │   ├── chat/       # Real-time chat
│   │   └── ...
│   ├── components/     # Shared UI components
│   │   ├── Navigation.tsx
│   │   ├── UI.tsx
│   │   ├── Toast.tsx
│   │   ├── auth.tsx
│   │   └── api.ts      # Typed API client
│   └── lib/            # Utility helpers
├── public/             # Static assets
├── Dockerfile          # Multi-stage production image
├── next.config.mjs     # Next.js config (rewrites, standalone)
├── tailwind.config.js
└── package.json
```

---

## Scripts

| Command | Description |
|---|---|
| `bun run dev` | Start dev server with hot reload |
| `bun run build` | Production build |
| `bun run start` | Start production server |
| `bun run lint` | ESLint check |
| `bun audit` | Security audit (0 vulnerabilities) |

---

## Related Docs

- [Getting Started](../docs/guides/GETTING_STARTED.md)
- [Running Frontend & Backend](../docs/guides/RUNNING_FRONTEND_BACKEND.md)
- [Docker Setup](../docs/guides/DOCKER_SETUP.md)
- [DSSC Theory](../docs/architecture/DSSC_THEORY_AND_MATHEMATICAL_SPECIFICATION.md)
