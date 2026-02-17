# tests/conftest.py
"""
Pytest configuration and shared fixtures for DCASS tests.
"""

import pytest
import sys
from pathlib import Path


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires indices)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection.
    
    - Skip integration tests if --no-integration flag is passed
    - Add slow marker to integration tests
    """
    skip_integration = pytest.mark.skip(reason="--no-integration flag passed")
    
    for item in items:
        # Mark integration tests as slow
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Skip integration tests if flag passed
        if config.getoption("--no-integration", default=False):
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-integration",
        action="store_true",
        default=False,
        help="Skip integration tests that require indices"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


# ============================================================
# Shared Fixtures
# ============================================================

@pytest.fixture(scope="session")
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def indices_path(project_root):
    """Return path to indices directory."""
    return project_root / "data" / "indices"


@pytest.fixture(scope="session")
def indices_exist(indices_path):
    """Check if all indices exist."""
    required = ["image.index", "text.index", "audio.index"]
    return all((indices_path / f).exists() for f in required)
