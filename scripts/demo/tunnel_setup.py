#!/usr/bin/env python3
"""
DCASS Public Port Forwarding & Evaluation Tunnel.
Exposes the Next.js frontend (which proxies backend API requests) to the public internet
via ngrok or localtunnel so evaluators and panel members can test the demo from any device.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RUN_DIR = PROJECT_ROOT / "storage" / "run"
TUNNEL_PID_FILE = RUN_DIR / "tunnel.pid"
TUNNEL_LOG_FILE = RUN_DIR / "tunnel.log"

from scripts.demo.lan_demo_setup import (
    check_port_open,
    is_pid_alive,
    print_ascii_qr,
    start_backend_if_needed,
    start_frontend_if_needed,
)


def stop_tunnel():
    """Stop active public tunnel."""
    if TUNNEL_PID_FILE.exists():
        try:
            pid = int(TUNNEL_PID_FILE.read_text().strip())
            if is_pid_alive(pid):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                print(f"[*] Stopped public tunnel process (PID {pid}).")
            else:
                print(f"[*] Tunnel PID {pid} is no longer active.")
        except Exception as e:
            print(f"[!] Warning stopping tunnel: {e}")
        finally:
            if TUNNEL_PID_FILE.exists():
                TUNNEL_PID_FILE.unlink()
        print("[+] Tunnel stopped.")
    else:
        print("[*] No active tunnel found.")


def has_ngrok_token() -> bool:
    """Check if an ngrok authtoken is configured anywhere."""
    if os.environ.get("NGROK_AUTHTOKEN"):
        return True
    for cfg_path in [
        Path.home() / ".config" / "ngrok" / "ngrok.yml",
        Path.home() / ".ngrok2" / "ngrok.yml",
    ]:
        if cfg_path.exists():
            try:
                content = cfg_path.read_text()
                if "authtoken:" in content:
                    return True
            except Exception:
                pass
    return False


def find_ngrok_bin() -> str:
    """Find the installed ngrok binary."""
    for candidate in [
        Path("/home/img1/.local/bin/ngrok"),
        PROJECT_ROOT / ".venv" / "bin" / "ngrok",
        Path.home() / "bin" / "ngrok",
    ]:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    import shutil
    return shutil.which("ngrok") or "ngrok"


def start_ngrok_tunnel(port: int = 3000, authtoken: str = None) -> str:
    """Start ngrok tunnel as an independent background daemon."""
    ngrok_bin = find_ngrok_bin()

    token = authtoken or os.environ.get("NGROK_AUTHTOKEN")
    if token:
        subprocess.run([ngrok_bin, "config", "add-authtoken", token], capture_output=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [ngrok_bin, "http", str(port), "--log=stdout"]

    with open(TUNNEL_LOG_FILE, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,
        )

    TUNNEL_PID_FILE.write_text(str(proc.pid))

    import urllib.request
    import json

    public_url = None
    for _ in range(30):
        time.sleep(0.5)
        if proc.poll() is not None:
            log_content = TUNNEL_LOG_FILE.read_text() if TUNNEL_LOG_FILE.exists() else ""
            if "ERR_NGROK_4018" in log_content or "authentication failed" in log_content:
                print("\n[!] Ngrok authentication failed. Run: ngrok config add-authtoken <token>")
            else:
                print(f"\n[!] Ngrok exited unexpectedly: {log_content[:200]}")
            return None

        try:
            req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                tunnels = data.get("tunnels", [])
                if tunnels:
                    for t in tunnels:
                        p_url = t.get("public_url", "")
                        if p_url.startswith("https://"):
                            public_url = p_url
                            break
                        elif p_url and not public_url:
                            public_url = p_url
                    if public_url:
                        break
        except Exception:
            pass

    return public_url


def start_localtunnel(port: int = 3000) -> str:
    """Start localtunnel via npx for instant public URL without accounts/tokens."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["npx", "--yes", "localtunnel", "--port", str(port)]

    with open(TUNNEL_LOG_FILE, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp,
            text=True,
            bufsize=1,
        )

    TUNNEL_PID_FILE.write_text(str(proc.pid))

    public_url = None
    # Read the initial output line from localtunnel to capture URL
    start_time = time.time()
    while time.time() - start_time < 15:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.2)
            continue
        with open(TUNNEL_LOG_FILE, "a") as log_f:
            log_f.write(line)
        match = re.search(r"https?://[^\s]+", line)
        if match:
            public_url = match.group(0).strip()
            break

    if not public_url:
        print("[!] Could not detect localtunnel public URL. Check storage/run/tunnel.log")
        return None

    return public_url


def main():
    parser = argparse.ArgumentParser(description="DCASS Public Tunnel Setup Utility")
    parser.add_argument(
        "--provider",
        choices=["auto", "ngrok", "localtunnel"],
        default="auto",
        help="Tunnel provider: ngrok (requires token) or localtunnel (instant, no signup)",
    )
    parser.add_argument("--authtoken", type=str, default=None, help="Ngrok authtoken")
    parser.add_argument("--port", type=int, default=3000, help="Local frontend port to expose")
    parser.add_argument("--stop", action="store_true", help="Stop any active tunnel")
    args = parser.parse_args()

    if args.stop:
        stop_tunnel()
        return

    print("=" * 72)
    print("      DCASS EXECUTIVE PUBLIC TUNNEL & EVALUATION SHARING")
    print("=" * 72)

    # 1. Ensure Local Stack is running
    print("[*] Checking local services...")
    start_backend_if_needed(api_port=8000)
    start_frontend_if_needed(frontend_port=args.port)

    # 2. Establish Tunnel
    public_url = None
    provider_used = ""

    token = args.authtoken or os.environ.get("NGROK_AUTHTOKEN")

    if args.provider == "ngrok" or (args.provider == "auto" and (token or has_ngrok_token())):
        print("[*] Initiating secure ngrok public tunnel...")
        try:
            public_url = start_ngrok_tunnel(port=args.port, authtoken=token)
            if public_url:
                provider_used = "Ngrok Secure Tunnel"
        except Exception as e:
            print(f"[!] Ngrok failed: {e}")

    if not public_url:
        print("[*] Initiating zero-config instant localtunnel...")
        public_url = start_localtunnel(port=args.port)
        if public_url:
            provider_used = "Localtunnel Public Gateway"

    if not public_url:
        print("\n[!] Failed to establish a public tunnel. Check internet connection and logs.")
        print("    storage/run/tunnel.log")
        print("=" * 72)
        sys.exit(1)

    eval_url = f"{public_url}/presentation"
    login_url = f"{public_url}/login"

    print("\n" + "=" * 72)
    print(f"      PUBLIC LIVE URL GENERATED VIA {provider_used.upper()}")
    print("=" * 72)
    print(f"  * Presentation Dashboard (Share with Panel / External Evaluators):")
    print(f"    {eval_url}")
    print(f"\n  * Direct Login URL:")
    print(f"    {login_url}")
    print("-" * 72)
    print("SCAN THIS QR CODE ON ANY PHONE / DEVICE (Works on cellular/external internet):")
    print_ascii_qr(eval_url)
    print("-" * 72)
    print("DEMO CREDENTIALS FOR PANEL EVALUATION:")
    print("  - Username: alice    Password: alice12345 (Sender)")
    print("  - Username: bob      Password: bob12345   (Receiver)")
    print("  - Username: charlie  Password: charlie12345")
    print("\nTo stop this public tunnel when finished:")
    print("  make tunnel-stop")
    print("=" * 72)


if __name__ == "__main__":
    main()
