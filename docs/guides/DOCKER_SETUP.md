# DCASS Docker deployment and simulation guide

This guide covers building and running the DCASS backend, Next.js frontend, Caddy HTTPS proxy, and CLI simulation sandbox with Docker Compose.

## Prerequisites

- Docker Engine and Docker Compose v2.
- Prebuilt FAISS indices in `storage/data/indices/` (around 575 MB) and runtime checkpoints in `storage/models/` (around 85 MB for `gan_generator.pt`, `warden.pt`, `rl_agent.pt`, and `tpp_ppo_*.pt`). Run `python scripts/data/download_hf_indices.py` to fetch indices if missing.
- Optional media previews require `storage/data/raw/` and `storage/data/audio/`.
- At least 4 GB RAM free for the backend container (6 GB configured limit when loading PyTorch, CLIP, and FAISS on CPU).

## Environment configuration

Copy [.env.example](../../.env.example) to `.env` before deploying outside localhost:

```bash
cp .env.example .env
```

Key settings in `.env`:

- `DCASS_ENV`: Set to `production` when exposing the stack on a server.
- `DCASS_JWT_SECRET`: Required in production. Set to a random 32+ character secret.
- `DCASS_API_TOKEN`: Required in production for non-JWT script access; logged-in browser users authenticate with their JWT.
- `DCASS_CORS_ORIGINS`: Comma-separated frontend origins allowed by FastAPI.
- `NEXT_PUBLIC_API_URL`: Browser-visible backend origin baked at frontend build time (for example `http://localhost:8000` locally, or omit when serving `/api/*` behind Caddy on port 443).
- `DOMAIN`: Domain name for automatic Let's Encrypt TLS certificates when running the `proxy` profile.

## Running the web stack (local or server)

Build and start the backend API (`:8000`) and Next.js frontend (`:3001`):

```bash
make docker-web
# equivalent: docker compose --profile web up -d --build
```

The backend runs with a single Uvicorn worker (`--workers 1`) because active transmission state and locks live in process memory. User accounts persist in SQLite at `/app/storage/db/auth.db` inside the `dcass-db` Docker volume, and transmitted wire packets live under `./storage/shared_channel`.

Stop the web stack without deleting the database volume:

```bash
make docker-web-down
```

## VPS deployment with automatic HTTPS (Caddy)

Point your DNS A/AAAA record to the server, set `DOMAIN`, `DCASS_ENV=production`, `DCASS_JWT_SECRET`, `DCASS_API_TOKEN`, and `DCASS_CORS_ORIGINS=https://your-domain.com` in `.env`, then start the web and proxy profiles:

```bash
make docker-proxy
# equivalent: docker compose --profile web --profile proxy up -d --build
```

[deploy/Caddyfile](../../deploy/Caddyfile) routes `/api/*` to `dcass-api:8000` and all other requests to `dcass-frontend:3001` over TLS on ports 80 and 443. [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts) automatically sends same-origin `/api/*` requests when loaded over HTTPS on port 443.

## Running the CLI simulation sandbox

Run the standalone sender (`dcass-alice`) and receiver (`dcass-bob`) containers sharing `./storage/shared_channel`:

```bash
docker compose build
docker compose up --abort-on-container-exit dcass-sender dcass-receiver
```

Override transmission parameters via environment variables:

```bash
DCASS_MODE=rl DCASS_MESSAGE="Test payload" docker compose up --abort-on-container-exit dcass-sender dcass-receiver
```

## Training and TensorBoard profiles

Run model training or TensorBoard monitoring using Compose profiles:

```bash
docker compose --profile training run --rm dcass-train-gan
docker compose --profile training run --rm dcass-train-rl
docker compose --profile monitoring up tensorboard
```

TensorBoard is served on port `6006` reading `./storage/logs`.

## Troubleshooting

- **Backend container fails health check with `DCASS_JWT_SECRET` or `503` errors**: Confirm `DCASS_ENV=production` has both `DCASS_JWT_SECRET` and `DCASS_API_TOKEN` set in `.env`, then recreate `dcass-api`.
- **Frontend `/api/*` returns 502**: `dcass-frontend` waits for `dcass-api` to pass `/api/health`. Check `docker compose logs dcass-api` to verify CLIP and FAISS finished loading within the 60s start period.
- **Permission errors on bind mounts**: If you uncomment `USER dcass` in [Dockerfile](../../Dockerfile), ensure host ownership on `./storage/shared_channel` matches the container user UID.
