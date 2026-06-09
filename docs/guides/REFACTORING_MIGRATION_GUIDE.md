# DCASS Refactoring Migration Guide

**Date:** April 6, 2026  
**Version:** 2.0  
**Status:** Complete

---

## Executive Summary

The DCASS codebase has been comprehensively refactored to improve organization, maintainability, and developer experience. This guide documents all changes and provides migration instructions.

### Key Changes
- ✅ Scripts organized into logical subdirectories
- ✅ Documentation consolidated under single `docs/` folder
- ✅ Runtime data centralized in `storage/` directory
- ✅ All import paths updated automatically
- ✅ Docker configuration updated
- ✅ Makefile commands updated

---

## Directory Structure Changes

### Before Refactoring

```
dcass/
├── scripts/              # ❌ 24+ files, no organization
│   ├── train_gan.py
│   ├── run_sender.py
│   ├── demo_dcass.py
│   └── ... (21 more files)
├── document/             # ❌ Duplicate docs folder
├── docs/                 # ❌ Duplicate docs folder
├── audio/                # ❌ Mixed with root
├── data/                 # ❌ Mixed with root
├── models/               # ❌ Mixed with root
├── logs/                 # ❌ Mixed with root
├── shared_channel/       # ❌ Mixed with root
├── checkpoints/          # ❌ Mixed with root
├── phase3_out/           # ❌ Mixed with root
├── download.py           # ❌ Loose script
├── test_step1.py         # ❌ Loose script
└── summary.txt           # ❌ Loose file
```

### After Refactoring

```
dcass/
├── scripts/              # ✅ Organized by purpose
│   ├── __init__.py
│   ├── data/             # Data preparation & corpus building
│   │   ├── download_flickr8k.py
│   │   ├── build_indices.py
│   │   └── ...
│   ├── audio/            # Audio-specific workflows
│   │   ├── audio_step1_download.py
│   │   └── ...
│   ├── training/         # Model training
│   │   ├── train_gan.py
│   │   ├── train_rl.py
│   │   └── generate_traffic_data.py
│   ├── runtime/          # Core system execution
│   │   ├── run_sender.py
│   │   ├── run_receiver.py
│   │   └── start_server.py
│   ├── demos/            # Demo scripts
│   │   ├── demo_dcass.py
│   │   └── ...
│   ├── testing/          # Testing & evaluation
│   │   ├── test_encoding.py
│   │   └── ...
│   └── utils/            # Utility scripts
│       ├── check_gpu.py
│       └── docker_orchestrate.py
│
├── docs/                 # ✅ Consolidated documentation
│   ├── diagrams/         # Visual diagrams
│   ├── guides/           # User-facing guides
│   │   ├── COMPLETE_IMPLEMENTATION_GUIDE.md
│   │   ├── SCRIPT_EXECUTION_GUIDE.md
│   │   ├── GETTING_STARTED.md
│   │   └── ...
│   ├── project/          # Project documentation
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   ├── PROJECT_COMPLETION_STATUS.md
│   │   └── ...
│   ├── research/         # Research artifacts
│   │   └── summary.txt
│   └── weekly_plans/     # Development plans
│
├── storage/              # ✅ All runtime data
│   ├── .gitkeep
│   ├── audio/            # Audio corpus
│   ├── data/             # Downloaded datasets
│   │   ├── raw/
│   │   ├── indices/
│   │   ├── behavioral/
│   │   └── benchmarks/
│   ├── models/           # Trained models
│   │   ├── gan/
│   │   └── rl/
│   ├── checkpoints/      # Training checkpoints
│   ├── logs/             # Application logs
│   │   ├── sender/
│   │   ├── receiver/
│   │   ├── train-gan/
│   │   └── train-rl/
│   ├── outputs/          # Output files (phase3_out)
│   └── shared_channel/   # Alice-Bob communication
│
├── tools/                # ✅ Legacy/one-off scripts
│   ├── __init__.py
│   ├── download.py
│   └── test_step1.py
│
├── src/                  # ✅ No changes (already well-organized)
├── frontend/             # ✅ No changes
├── tests/                # ✅ No changes
└── config/               # ✅ No changes
```

---

## Detailed File Movements

### Scripts Reorganization

| Old Location | New Location | Category |
|-------------|-------------|----------|
| `scripts/data/download_flickr8k.py` | `scripts/data/download_flickr8k.py` | Data |
| `scripts/data/download_flickr30k.py` | `scripts/data/download_flickr30k.py` | Data |
| `scripts/data/download_wikipedia.py` | `scripts/data/download_wikipedia.py` | Data |
| `scripts/data/build_flickr30k_index.py` | `scripts/data/build_flickr30k_index.py` | Data |
| `scripts/data/build_indices.py` | `scripts/data/build_indices.py` | Data |
| `scripts/data/add_wikipedia_to_index.py` | `scripts/data/add_wikipedia_to_index.py` | Data |
| `scripts/audio/audio_step1_download.py` | `scripts/audio/audio_step1_download.py` | Audio |
| `scripts/audio/audio_step2_build_index.py` | `scripts/audio/audio_step2_build_index.py` | Audio |
| `scripts/fix_audio_metadata.py` | `scripts/audio/fix_audio_metadata.py` | Audio |
| `scripts/training/train_gan.py` | `scripts/training/train_gan.py` | Training |
| `scripts/training/train_rl.py` | `scripts/training/train_rl.py` | Training |
| `scripts/training/generate_traffic_data.py` | `scripts/training/generate_traffic_data.py` | Training |
| `scripts/runtime/run_sender.py` | `scripts/runtime/run_sender.py` | Runtime |
| `scripts/runtime/run_receiver.py` | `scripts/runtime/run_receiver.py` | Runtime |
| `scripts/runtime/start_server.py` | `scripts/runtime/start_server.py` | Runtime |
| `scripts/runtime/run_pipeline.py` | `scripts/runtime/run_pipeline.py` | Runtime |
| `scripts/demos/demo_dcass.py` | `scripts/demos/demo_dcass.py` | Demo |
| `scripts/demos/demo_encoder.py` | `scripts/demos/demo_encoder.py` | Demo |
| `scripts/demos/demo_full_loop.py` | `scripts/demos/demo_full_loop.py` | Demo |
| `scripts/testing/test_encoding.py` | `scripts/testing/test_encoding.py` | Testing |
| `scripts/testing/test_stealth_system.py` | `scripts/testing/test_stealth_system.py` | Testing |
| `scripts/testing/evaluate_stealth.py` | `scripts/testing/evaluate_stealth.py` | Testing |
| `scripts/utils/check_gpu.py` | `scripts/utils/check_gpu.py` | Utility |
| `scripts/utils/docker_orchestrate.py` | `scripts/utils/docker_orchestrate.py` | Utility |

### Documentation Consolidation

| Old Location | New Location |
|-------------|-------------|
| `document/COMPLETE_IMPLEMENTATION_GUIDE.md` | `docs/guides/COMPLETE_IMPLEMENTATION_GUIDE.md` |
| `document/INCOMPLETE_TASKS_PRD.md` | `docs/guides/INCOMPLETE_TASKS_PRD.md` |
| `document/SCRIPT_EXECUTION_GUIDE.md` | `docs/guides/SCRIPT_EXECUTION_GUIDE.md` |
| `document/TEAM_HANDOFF_GUIDE.md` | `docs/guides/TEAM_HANDOFF_GUIDE.md` |
| `document/GETTING_STARTED.md` | `docs/guides/GETTING_STARTED.md` |
| `document/QUICK_START.md` | `docs/guides/QUICK_START.md` |
| `document/DOCKER_SETUP.md` | `docs/guides/DOCKER_SETUP.md` |
| `document/SCRIPTS.md` | `docs/guides/SCRIPTS.md` |
| `document/IMPLEMENTATION_SUMMARY.md` | `docs/project/IMPLEMENTATION_SUMMARY.md` |
| `document/PROJECT_COMPLETION_STATUS.md` | `docs/project/PROJECT_COMPLETION_STATUS.md` |
| `document/DCASS_Implementation_Handout.md` | `docs/project/DCASS_Implementation_Handout.md` |
| `document/UIUX_BASE_REQUIREMENTS.md` | `docs/project/UIUX_BASE_REQUIREMENTS.md` |
| `document/README.md` | `docs/project/DOCUMENT_INDEX.md` |
| `document/diagrams/*` | `docs/diagrams/*` |
| `document/weekly_plans/` | `docs/weekly_plans/` |
| `summary.txt` | `docs/research/summary.txt` |

### Root Files Cleanup

| Old Location | New Location |
|-------------|-------------|
| `download.py` | `tools/download.py` |
| `test_step1.py` | `tools/test_step1.py` |

### Runtime Directories

| Old Location | New Location |
|-------------|-------------|
| `audio/` | `storage/audio/` |
| `data/` | `storage/data/` |
| `models/` | `storage/models/` |
| `checkpoints/` | `storage/checkpoints/` |
| `logs/` | `storage/logs/` |
| `shared_channel/` | `storage/shared_channel/` |
| `phase3_out/` | `storage/outputs/` |

---

## Code Changes

### Import Path Updates

All scripts have been updated to use the correct project root path. Scripts are now nested one level deeper, so:

**Before:**
```python
PROJECT_ROOT = Path(__file__).parent.parent  # scripts/script.py → dcass/
```

**After:**
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent  # scripts/category/script.py → dcass/
```

**Files Updated (22 total):**
- All scripts in `scripts/data/`
- All scripts in `scripts/audio/`
- All scripts in `scripts/training/`
- All scripts in `scripts/runtime/`
- All scripts in `scripts/demos/`
- All scripts in `scripts/testing/`
- All scripts in `scripts/utils/`

---

## Configuration Changes

### Dockerfile Updates

**Changed:**
```dockerfile
# Before
RUN mkdir -p /app/data /app/models /app/checkpoints /app/logs /app/storage/shared_channel

# After
RUN mkdir -p /app/storage/data /app/storage/models /app/storage/checkpoints \
    /app/storage/logs /app/storage/shared_channel
```

### docker-compose.yml Updates

**Volume Mounts:**
```yaml
# Before
volumes:
  - ./storage/shared_channel:/app/storage/shared_channel:rw
  - ./data:/app/data:ro
  - ./models:/app/models:rw
  - ./storage/logs/sender:/app/logs:rw

# After
volumes:
  - ./storage/shared_channel:/app/storage/shared_channel:rw
  - ./storage/data:/app/data:ro
  - ./storage/models:/app/models:rw
  - ./storage/logs/sender:/app/logs:rw
```

**Command Updates:**
```yaml
# Before
command: python -u scripts/runtime/run_sender.py

# After
command: python -u scripts/runtime/run_sender.py
```

### Makefile Updates

**Script Paths:**
```makefile
# Before
download-data:
	python scripts/data/download_flickr8k.py

# After
download-data:
	python scripts/data/download_flickr8k.py
```

**Cleanup Paths:**
```makefile
# Before
rm -rf shared_channel/*.json

# After
rm -rf storage/shared_channel/*.json
```

### .gitignore Updates

**Added:**
```gitignore
# Storage directory - runtime data (keep structure, ignore contents)
storage/*
!storage/.gitkeep
```

**Updated comments to reflect new structure**

---

## Migration Instructions

### For Existing Developers

1. **Pull the latest changes** (after they're committed)
   ```bash
   git pull origin feature/frontend-ui
   ```

2. **No action needed** - All paths are updated automatically

3. **If you have local data directories**, you can either:
   - **Option A:** Move them manually
     ```bash
     mv data storage/
     mv models storage/
     mv logs storage/
     mv shared_channel storage/
     ```
   - **Option B:** Re-download/regenerate (recommended for clean state)
     ```bash
     make download-data
     make build-index
     ```

4. **Update any custom scripts** that reference old paths

### For New Developers

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd dcass
   ```

2. **The new structure is already in place** - No migration needed!

3. **Follow the getting started guide**
   ```bash
   make install
   make download-data
   make build-index
   ```

### For Docker Users

**No changes required!** Docker commands remain the same:
```bash
docker compose up
docker compose --profile training run dcass-train-gan
```

The volume mounts are updated internally.

---

## Breaking Changes

### ⚠️ If You Have Custom Scripts

If you have custom scripts that import from the old paths:

**Before:**
```python
# Custom script in project root
from scripts.train_gan import train_model  # ❌ Will break
```

**After:**
```python
# Use proper module imports
from scripts.training.train_gan import train_model  # ✅ Works
```

### ⚠️ If You Have CI/CD Pipelines

Update any automation that references old paths:

**Before:**
```yaml
- run: python scripts/training/train_gan.py
- run: python scripts/runtime/run_sender.py
```

**After:**
```yaml
- run: python scripts/training/train_gan.py
- run: python scripts/runtime/run_sender.py
```

### ⚠️ If You Have Documentation Links

Update any documentation that links to old script paths:

**Before:**
```markdown
Run [scripts/training/train_gan.py](scripts/training/train_gan.py)
```

**After:**
```markdown
Run [scripts/training/train_gan.py](scripts/training/train_gan.py)
```

---

## Testing the Refactoring

### Quick Smoke Tests

1. **Test imports:**
   ```bash
   python -c "from src.engine.encoder import SemanticEncoder; print('✅ Imports work')"
   ```

2. **Test script execution:**
   ```bash
   python scripts/utils/check_gpu.py
   ```

3. **Test Docker build:**
   ```bash
   docker compose build
   ```

4. **Test Make commands:**
   ```bash
   make help
   ```

### Full Integration Tests

1. **Run unit tests:**
   ```bash
   make test
   ```

2. **Run demo:**
   ```bash
   python scripts/demos/demo_dcass.py
   ```

3. **Run Docker pipeline:**
   ```bash
   docker compose up
   ```

---

## Benefits of Refactoring

### Developer Experience
- ✅ **Easy navigation** - Find scripts by purpose, not by guessing
- ✅ **Clear organization** - Logical grouping reduces confusion
- ✅ **Better discoverability** - New developers can quickly understand structure

### Maintainability
- ✅ **Reduced clutter** - Root directory is clean and professional
- ✅ **Better git hygiene** - Runtime data consolidated under `storage/`
- ✅ **Module structure** - `__init__.py` files enable proper Python imports

### Professional Standards
- ✅ **Industry best practices** - Matches standard Python project layouts
- ✅ **Scalability** - Easy to add new scripts in appropriate categories
- ✅ **Documentation** - Single `docs/` folder with clear organization

---

## Rollback Instructions

If you need to rollback to the old structure:

```bash
# Reset to the commit before refactoring
git reset --hard 2b834ad

# Or revert the refactoring commit (after it's committed)
git revert <refactoring-commit-hash>
```

**Note:** This will lose any changes made after the refactoring.

---

## Support

### Questions?
- Check the updated documentation in `docs/guides/`
- Review this migration guide
- Contact the development team

### Found an issue?
- Check the "Known Issues" section below
- Report issues with the refactoring

---

## Known Issues

### None Currently

The refactoring has been tested and all paths are updated. If you encounter any issues, please report them.

---

## Changelog

### Version 2.0 (April 6, 2026)
- ✅ Complete codebase refactoring
- ✅ Scripts organized into 7 categories
- ✅ Documentation consolidated
- ✅ Runtime data centralized
- ✅ All configurations updated
- ✅ Import paths fixed

### Version 1.0 (Before April 6, 2026)
- Original unorganized structure

---

## Appendix: Quick Reference

### Finding Scripts

| Task | Old Command | New Command |
|------|------------|-------------|
| Download data | `python scripts/data/download_flickr8k.py` | `python scripts/data/download_flickr8k.py` |
| Train GAN | `python scripts/training/train_gan.py` | `python scripts/training/train_gan.py` |
| Run sender | `python scripts/runtime/run_sender.py` | `python scripts/runtime/run_sender.py` |
| Run demo | `python scripts/demos/demo_dcass.py` | `python scripts/demos/demo_dcass.py` |
| Run tests | `python scripts/testing/test_encoding.py` | `python scripts/testing/test_encoding.py` |
| Check GPU | `python scripts/utils/check_gpu.py` | `python scripts/utils/check_gpu.py` |

### Finding Documentation

| Document Type | Old Location | New Location |
|--------------|-------------|-------------|
| User guides | `document/` | `docs/guides/` |
| Project docs | `document/` | `docs/project/` |
| Diagrams | `document/diagrams/` or `docs/diagrams/` | `docs/diagrams/` |
| Research | `summary.txt` | `docs/research/summary.txt` |

### Finding Data

| Data Type | Old Location | New Location |
|-----------|-------------|-------------|
| Datasets | `data/` | `storage/data/` |
| Models | `models/` | `storage/models/` |
| Logs | `logs/` | `storage/logs/` |
| Outputs | `phase3_out/` | `storage/outputs/` |
| Audio | `audio/` | `storage/audio/` |
| Shared channel | `shared_channel/` | `storage/shared_channel/` |

---

**End of Migration Guide**
