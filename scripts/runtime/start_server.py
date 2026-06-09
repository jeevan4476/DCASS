#!/usr/bin/env python3
# scripts/start_server.py
"""
Start the DCASS FastAPI backend server.

This script starts the FastAPI server without Docker for local development.

Usage:
    python scripts/start_server.py
    python scripts/start_server.py --port 8000 --reload
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Start DCASS FastAPI backend server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--no-warmup", action="store_true", help="Skip model warmup on startup")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 Starting DCASS FastAPI Backend Server")
    print("=" * 70)
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"📖 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔄 Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print("=" * 70)
    print()
    
    # Import uvicorn here to avoid issues if not installed
    try:
        import uvicorn
    except ImportError:
        print("❌ Error: uvicorn not installed")
        print("Install with: pip install uvicorn")
        sys.exit(1)

    # Pre-warm the engine unless disabled
    if not args.no_warmup and not args.reload:
        # Import and call warmup
        from src.api.server import warmup
        warmup()
    elif args.reload:
        print("⚠️  Warmup skipped (auto-reload enabled)\n")
    else:
        print("⚠️  Warmup skipped (--no-warmup flag)\n")

    # Start server
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
