#!/usr/bin/env python3
"""
DCASS Dual-Laptop & Mobile Presentation Setup & Automation Utility.
Detects local network IP, verifies port accessibility, starts background
servers bound to 0.0.0.0 for LAN/mobile testing, and displays QR codes.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_DIR = PROJECT_ROOT / "storage" / "run"
BACKEND_PID_FILE = RUN_DIR / "lan_backend.pid"
FRONTEND_PID_FILE = RUN_DIR / "lan_frontend.pid"
BACKEND_LOG_FILE = RUN_DIR / "lan_backend.log"
FRONTEND_LOG_FILE = RUN_DIR / "lan_frontend.log"


def get_local_ip() -> str:
    """Detect the active local network IPv4 address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def check_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Verify whether a given TCP port is accepting connections."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def is_pid_alive(pid: int) -> bool:
    """Check if process with PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_backend_if_needed(api_port: int = 8000) -> bool:
    """Start backend API on 0.0.0.0:8000 if not already running."""
    if check_port_open("127.0.0.1", api_port):
        print(f"[*] Backend already running on port {api_port}.")
        return True

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    print(f"[*] Starting backend on 0.0.0.0:{api_port} ...")
    cmd = [
        str(python_bin),
        "-m",
        "uvicorn",
        "src.api.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(api_port),
    ]

    with open(BACKEND_LOG_FILE, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,
        )

    BACKEND_PID_FILE.write_text(str(proc.pid))

    # Wait for ready
    print("   Waiting for backend engine warmup...", end="", flush=True)
    for _ in range(60):
        time.sleep(0.5)
        print(".", end="", flush=True)
        if check_port_open("127.0.0.1", api_port):
            print(" Ready!")
            return True

    print(" Failed to start backend within 30s. Check logs at storage/run/lan_backend.log")
    return False


def start_frontend_if_needed(frontend_port: int = 3000, dev_mode: bool = False) -> bool:
    """Start frontend on 0.0.0.0:3000 if not already running."""
    if check_port_open("127.0.0.1", frontend_port):
        print(f"[*] Frontend already running on port {frontend_port}.")
        return True

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    frontend_dir = PROJECT_ROOT / "frontend"

    has_prod_build = (frontend_dir / ".next" / "BUILD_ID").exists()
    mode_label = "development (hot-reload)" if (dev_mode or not has_prod_build) else "production"
    print(f"[*] Starting frontend in {mode_label} mode on 0.0.0.0:{frontend_port} ...")
    if dev_mode or not has_prod_build:
        cmd = ["npx", "next", "dev", "-p", str(frontend_port), "-H", "0.0.0.0"]
    else:
        cmd = ["npx", "next", "start", "-p", str(frontend_port), "-H", "0.0.0.0"]

    with open(FRONTEND_LOG_FILE, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(frontend_dir),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,
        )

    FRONTEND_PID_FILE.write_text(str(proc.pid))

    print("   Waiting for frontend web server...", end="", flush=True)
    for _ in range(30):
        time.sleep(0.5)
        print(".", end="", flush=True)
        if check_port_open("127.0.0.1", frontend_port):
            print(" Ready!")
            return True

    print(" Failed to start frontend. Check logs at storage/run/lan_frontend.log")
    return False


def stop_lan_services():
    """Stop background processes started for LAN demo."""
    stopped = 0
    for name, pid_file in [("Backend", BACKEND_PID_FILE), ("Frontend", FRONTEND_PID_FILE)]:
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if is_pid_alive(pid):
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    print(f"[*] Stopped {name} (PID {pid}).")
                    stopped += 1
            except Exception as e:
                print(f"[!] Warning stopping {name}: {e}")
            finally:
                if pid_file.exists():
                    pid_file.unlink()
    if stopped == 0:
        print("[*] No active background LAN services found to stop.")
    else:
        print("[+] All LAN demo services stopped successfully.")


def print_ascii_qr(url: str):
    """Print an ASCII QR code for scanning on mobile phones."""
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        # Fallback if qrcode module encounters issues
        pass


def main():
    parser = argparse.ArgumentParser(description="DCASS LAN & Mobile Setup Helper")
    parser.add_argument(
        "--role",
        choices=["host", "client"],
        default="host",
        help="Machine role: host (Laptop 1) or client (Laptop 2 / Mobile)",
    )
    parser.add_argument("--host-ip", type=str, default=None, help="Host IP address (when testing client connectivity)")
    parser.add_argument("--full", action="store_true", help="Automatically start backend and frontend on 0.0.0.0")
    parser.add_argument("--start", action="store_true", help="Alias for --full")
    parser.add_argument("--dev", action="store_true", help="Start frontend in development hot-reload mode (next dev)")
    parser.add_argument("--stop", action="store_true", help="Stop background LAN services")
    args = parser.parse_args()

    if args.stop:
        stop_lan_services()
        return

    local_ip = get_local_ip()
    api_port = 8000
    frontend_port = 3000

    print("=" * 72)
    print("      DCASS EXECUTIVE DUAL-LAPTOP & MOBILE DEMO AUTOMATION")
    print("=" * 72)

    if args.role == "host":
        print(f"[+] Detected Host LAN IP: {local_ip}")

        # Start servers automatically if --full or --start requested
        if args.full or args.start or args.dev:
            print("\n[*] Initializing Automated LAN Stack (0.0.0.0)...")
            start_backend_if_needed(api_port)
            start_frontend_if_needed(frontend_port, dev_mode=args.dev)

        backend_active = check_port_open("127.0.0.1", api_port)
        frontend_active = check_port_open("127.0.0.1", frontend_port)

        print("\nService Status:")
        print(f"  - Port {api_port} (Backend API): {'ONLINE (0.0.0.0)' if backend_active else 'OFFLINE'}")
        print(f"  - Port {frontend_port} (Frontend UI): {'ONLINE (0.0.0.0)' if frontend_active else 'OFFLINE'}")

        if not (backend_active and frontend_active):
            print("\n[!] Services are not running yet. Run the following command to start both automatically:")
            print("    make lan-setup full")
            print("=" * 72)
            return

        mobile_url = f"http://{local_ip}:{frontend_port}/presentation?remote=http://{local_ip}:{api_port}"
        local_url = f"http://localhost:{frontend_port}/presentation"

        print("-" * 72)
        print("ACCESS URLS:")
        print(f"  * Laptop 1 (Alice / Host):")
        print(f"    {local_url}")
        print(f"\n  * Mobile Phone or Laptop 2 (Bob / Client):")
        print(f"    {mobile_url}")
        print("-" * 72)

        print("\nSCAN THIS QR CODE ON YOUR MOBILE (Must be connected to same Wi-Fi):")
        print_ascii_qr(mobile_url)

        print("-" * 72)
        print("HOW TO TEST FROM YOUR MOBILE:")
        print("1. Ensure your phone is connected to the same Wi-Fi network.")
        print("2. Open your Camera app or QR Scanner and point it at the code above.")
        print("3. Tap the link to open the presentation dashboard directly on your phone.")
        print("\nTo stop background servers when finished:")
        print("  make lan-stop")
        print("=" * 72)

    else:
        target = args.host_ip or local_ip
        print(f"[*] Validating connectivity to Laptop 1 at {target}...")
        backend_active = check_port_open(target, api_port)
        frontend_active = check_port_open(target, frontend_port)
        if backend_active and frontend_active:
            print("  [SUCCESS] All services reachable over LAN! You are ready for the presentation.")
        else:
            if not backend_active:
                print(f"  [ERROR] Port {api_port} on {target} is unreachable. Check firewall and hotspot settings.")
            if not frontend_active:
                print(f"  [ERROR] Port {frontend_port} on {target} is unreachable. Ensure frontend is running.")
        print("=" * 72)


if __name__ == "__main__":
    main()
