# DCASS Makefile
# Common commands for development workflow

.PHONY: install install-clip test lint format clean download-data build-index run help
.PHONY: docker-build docker-send docker-train docker-pipeline docker-clean docker-gen-data

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
