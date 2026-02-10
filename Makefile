# DCASS Makefile
# Common commands for development workflow

.PHONY: install install-clip test lint format clean download-data build-index run help

# Default target
help:
	@echo "DCASS - Dynamic Context-Aware Semantic Steganography"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install dependencies from requirements.txt"
	@echo "  make install-clip  - Install OpenAI CLIP (required for image encoding)"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Remove cache files and build artifacts"
	@echo "  make download-data - Download Flickr8k dataset"
	@echo "  make build-index   - Build FAISS indices"
	@echo "  make run           - Run the CLI"
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
	python scripts/download_flickr8k.py

# Build FAISS indices
build-index:
	python scripts/build_indices.py

# Build only image index
build-image-index:
	python scripts/build_indices.py --modality image

# Build only text index
build-text-index:
	python scripts/build_indices.py --modality text

# Run CLI
run:
	python -m src.cli.main

# Run encoding demo
demo-encode:
	python -m src.cli.main encode "a dog running through water"

# Run distribution demo
demo-distribute:
	python -m src.cli.main distribute "a dog running through water" casual
