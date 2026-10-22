# DCASS Makefile
# Common commands for development workflow
#
# Local server targets:
#   make server     Backend API (warmed engine, no reload) on :8000
#   make frontend   Production frontend build, served on :3001
#   make stack      Both in the background; logs under storage/run/
#   make stop       Stop the background stack
#
# Override settings on the command line, e.g.:
#   make server API_PORT=8080
#   make stack API_HOST=0.0.0.0 PUBLIC_API_URL=http://192.168.1.10:8000

.PHONY: help install install-clip test lint format clean download-data build-index run
.PHONY: check-venv venv server server-dev frontend frontend-install frontend-build frontend-check
.PHONY: frontend-dev stack stop logs
.PHONY: docker-build docker-send docker-train docker-pipeline docker-clean docker-gen-data

# Load local overrides from .env (gitignored; template in .env.example) and
# pass the DCASS_* / NEXT_PUBLIC_* settings through to child processes.
-include .env
export DCASS_DEVICE DCASS_CORS_ORIGINS DCASS_API_TOKEN
export DCASS_CONTEXT_SECRET DCASS_JWT_SECRET DCASS_ENV DCASS_GDRIVE_BASE_URL
export NEXT_PUBLIC_API_URL

# ---- Runtime configuration (all overridable) ------------------------------
VENV           ?= .venv
PY             ?= $(VENV)/bin/python
PIP            ?= $(VENV)/bin/pip

API_HOST       ?= 127.0.0.1
API_PORT       ?= 8000
FRONTEND_HOST  ?= 127.0.0.1
FRONTEND_PORT  ?= 3001
PUBLIC_API_URL ?= $(or $(NEXT_PUBLIC_API_URL),http://localhost:$(API_PORT))

RUN_DIR        ?= storage/run

# Default target
help:
	@echo "DCASS - Dynamic Context-Aware Semantic Steganography"
	@echo ""
	@echo "Local commands:"
	@echo "  make install         - Install dependencies from requirements.txt"
	@echo "  make install-clip    - Install OpenAI CLIP (required for image encoding)"
	@echo "  make test            - Run all tests"
	@echo "  make lint            - Run code linting"
	@echo "  make format          - Format code with black"
	@echo "  make clean           - Remove cache files and build artifacts"
	@echo "  make download-data   - Download Flickr8k dataset"
	@echo "  make build-index     - Build FAISS indices"
	@echo "  make run             - Run the CLI"
	@echo ""
	@echo "Local server commands:"
	@echo "  make server          - Backend API on :8000 (warmed engine, no reload)"
	@echo "  make server-dev      - Backend API with auto-reload"
	@echo "  make frontend        - Serve frontend on :3001 (production build)"
	@echo "  make frontend-dev    - Frontend dev server on :3000"
	@echo "  make frontend-build  - Force a production frontend rebuild"
	@echo "  make stack           - Backend + frontend in background (make stop / make logs)"
	@echo "  make venv            - Create .venv and install backend dependencies"
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-send     - Send sequence (auto: rl->gan->static fallback)"
	@echo "  make docker-gen-data - Generate synthetic human traffic data"
	@echo "  make docker-train    - Gen data + train GAN + train RL"
	@echo "  make docker-pipeline - Full pipeline: gen + train + send"
	@echo "  make docker-clean    - Stop and remove all containers"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Install CLIP separately (requires git)
install-clip:
	pip install git+https://github.com/openai/CLIP.git

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Run linting
lint:
	ruff check src/ tests/

# Format code
format:
	black src/ tests/ scripts/

# Clean cache and build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true

# Download datasets
download-data:
	python scripts/data/download_flickr8k.py

# Build FAISS indices
build-index:
	python scripts/data/build_indices.py

# Build only image index
build-image-index:
	python scripts/data/build_indices.py --modality image

# Build only text index
build-text-index:
	python scripts/data/build_indices.py --modality text

# Run CLI
run:
	python -m src.cli.main

# Run encoding demo
demo-encode:
	python -m src.cli.main encode "a dog running through water"

# Run distribution demo
demo-distribute:
	python -m src.cli.main distribute "a dog running through water" casual

# ═══════════════════════════════════════════════════════════════════════
# Local server targets (backend API + web frontend)
# ═══════════════════════════════════════════════════════════════════════

# Fail fast when the project virtualenv is missing
check-venv:
	@test -x $(PY) || { echo "ERROR: $(PY) not found. Create it with 'make venv' or set VENV=<path>."; exit 1; }

# Create the Python virtualenv and install backend dependencies
venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Backend API. Single worker on purpose: the engine singleton and the
# transmission session state live in this process, and each extra worker
# would load its own copy of the models and FAISS indices.
server: check-venv
	@echo "DCASS backend starting on http://$(API_HOST):$(API_PORT) (docs at /docs)"
	@exec $(PY) scripts/runtime/start_server.py --host $(API_HOST) --port $(API_PORT)

# Backend API with auto-reload on code changes
server-dev: check-venv
	@exec $(PY) scripts/runtime/start_server.py --host $(API_HOST) --port $(API_PORT) --reload

# Install frontend npm dependencies
frontend-install:
	@cd frontend && npm install --no-fund --no-audit

# Force a production frontend rebuild for the configured API URL
frontend-build: frontend-install
	@cd frontend && NEXT_PUBLIC_API_URL=$(PUBLIC_API_URL) NEXT_TELEMETRY_DISABLED=1 npm run build
	@printf '%s' '$(PUBLIC_API_URL)' > frontend/.next-api-url
	@echo "Frontend build ready (API URL: $(PUBLIC_API_URL))"

# Rebuild the frontend only when sources or the API URL changed
frontend-check:
	@if ! ( cd frontend \
		&& test -f .next/BUILD_ID \
		&& test "$$(cat .next-api-url 2>/dev/null)" = "$(PUBLIC_API_URL)" \
		&& test -z "$$(find src public next.config.mjs package.json -newer .next/BUILD_ID -type f 2>/dev/null | head -1)" ); then \
		echo "Frontend build missing or stale; rebuilding..."; \
		$(MAKE) --no-print-directory frontend-build; \
	else \
		echo "Frontend build is up to date."; \
	fi

# Production frontend (next start on $(FRONTEND_PORT))
frontend: frontend-install frontend-check
	@echo "DCASS frontend starting on http://$(FRONTEND_HOST):$(FRONTEND_PORT)"
	@cd frontend && exec env NODE_ENV=production NEXT_PUBLIC_API_URL=$(PUBLIC_API_URL) npm run start -- -p $(FRONTEND_PORT) -H $(FRONTEND_HOST)

# Frontend dev server with hot reload (:3000)
frontend-dev: frontend-install
	@echo "DCASS frontend (dev) starting on http://127.0.0.1:3000"
	@cd frontend && exec env NEXT_PUBLIC_API_URL=$(PUBLIC_API_URL) npm run dev

# Run backend and frontend together in the background
stack: check-venv frontend-install frontend-check
	@mkdir -p $(RUN_DIR)
	@for svc in backend frontend; do \
		if [ -f $(RUN_DIR)/$$svc.pid ] && kill -0 "$$(cat $(RUN_DIR)/$$svc.pid)" 2>/dev/null; then \
			echo "ERROR: $$svc already running (pid $$(cat $(RUN_DIR)/$$svc.pid)). Run 'make stop' first."; exit 1; \
		fi; \
	done
	@echo "Starting backend on http://$(API_HOST):$(API_PORT) ..."
	@( exec nohup $(PY) scripts/runtime/start_server.py --host $(API_HOST) --port $(API_PORT) > $(abspath $(RUN_DIR))/backend.log 2>&1 < /dev/null ) & echo $$! > $(RUN_DIR)/backend.pid
	@printf "Waiting for backend"
	@for i in $$(seq 1 300); do \
		if curl -sf http://127.0.0.1:$(API_PORT)/api/health > /dev/null 2>&1; then echo " ready."; break; fi; \
		printf "."; \
		if [ $$i -eq 300 ]; then \
			echo " timed out."; tail -n 40 $(RUN_DIR)/backend.log; \
			kill $$(cat $(RUN_DIR)/backend.pid) 2>/dev/null || true; rm -f $(RUN_DIR)/backend.pid; exit 1; \
		fi; \
		sleep 1; \
	done
	@echo "Starting frontend on http://$(FRONTEND_HOST):$(FRONTEND_PORT) ..."
	@( cd frontend && exec nohup env NODE_ENV=production NEXT_PUBLIC_API_URL=$(PUBLIC_API_URL) node_modules/.bin/next start -p $(FRONTEND_PORT) -H $(FRONTEND_HOST) > $(abspath $(RUN_DIR))/frontend.log 2>&1 < /dev/null ) & echo $$! > $(RUN_DIR)/frontend.pid
	@printf "Waiting for frontend"
	@for i in $$(seq 1 120); do \
		if curl -s -o /dev/null http://127.0.0.1:$(FRONTEND_PORT)/; then echo " ready."; break; fi; \
		printf "."; \
		if [ $$i -eq 120 ]; then \
			echo " timed out."; tail -n 40 $(RUN_DIR)/frontend.log; \
			kill $$(cat $(RUN_DIR)/frontend.pid) 2>/dev/null || true; rm -f $(RUN_DIR)/frontend.pid; exit 1; \
		fi; \
		sleep 1; \
	done
	@echo ""
	@echo "DCASS stack is running:"
	@echo "  Frontend: http://$(FRONTEND_HOST):$(FRONTEND_PORT)"
	@echo "  Backend:  http://$(API_HOST):$(API_PORT)/docs"
	@echo "  Logs:     make logs         Stop: make stop"

# Stop the background stack started with 'make stack'
stop:
	@for svc in backend frontend; do \
		if [ -f $(RUN_DIR)/$$svc.pid ]; then \
			pid="$$(cat $(RUN_DIR)/$$svc.pid)"; \
			if kill -0 "$$pid" 2>/dev/null; then \
				kill "$$pid" 2>/dev/null && echo "Stopped $$svc (pid $$pid)"; \
			else \
				echo "$$svc was not running (stale pid $$pid)"; \
			fi; \
			rm -f $(RUN_DIR)/$$svc.pid; \
		else \
			echo "$$svc: no pid file (not started by 'make stack')"; \
		fi; \
	done

# Tail the background stack logs
logs:
	@test -f $(RUN_DIR)/backend.log || { echo "No logs yet. Run 'make stack' first."; exit 1; }
	@tail -n 30 -f $(RUN_DIR)/backend.log $(RUN_DIR)/frontend.log

# ═══════════════════════════════════════════════════════════════════════
# Docker targets
# ═══════════════════════════════════════════════════════════════════════

# Build Docker images
docker-build:
	docker compose build

# Send sequence with auto fallback (rl → gan → static)
docker-send:
	docker compose up --abort-on-container-exit dcass-sender dcass-receiver

# Generate synthetic human traffic data
docker-gen-data:
	docker compose --profile training run --rm dcass-gen-traffic

# Run data generation + GAN + RL training
docker-train:
	docker compose --profile training run --rm dcass-gen-traffic
	docker compose --profile training run --rm dcass-train-gan
	docker compose --profile training run --rm dcass-train-rl

# Full end-to-end pipeline: gen → train → send
docker-pipeline:
	python scripts/utils/docker_orchestrate.py --full-pipeline

# Stop and remove all containers
docker-clean:
	docker compose down --remove-orphans
	@echo "Cleaning shared_channel..."
	rm -rf storage/shared_channel/*.json 2>/dev/null || true
