#!/usr/bin/env python3
"""
DCASS Log Archival & Rotation Utility.
Safely copies current and historical logs into a timestamped backup directory
under storage/logs_backup/, then truncates active service logs to 0 bytes so
running background servers maintain valid file descriptors.
"""

import argparse
import datetime
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
RUN_DIR = STORAGE_DIR / "run"
BACKUP_ROOT = STORAGE_DIR / "logs_backup"

ACTIVE_LOG_NAMES = {
    "lan_backend.log",
    "lan_frontend.log",
    "backend.log",
    "frontend.log",
    "tunnel.log",
}


def backup_and_clean_logs(backup_only: bool = False):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dest = BACKUP_ROOT / f"backup_{timestamp}"
    backup_dest.mkdir(parents=True, exist_ok=True)

    # Discover all log files under storage/run and storage/logs
    candidate_dirs = [RUN_DIR, STORAGE_DIR / "logs"]
    backed_up = []
    total_bytes = 0

    for c_dir in candidate_dirs:
        if not c_dir.exists():
            continue
        for root, _, files in os.walk(c_dir):
            for file in files:
                if file.endswith(".log"):
                    log_file = Path(root) / file
                    if log_file.is_file():
                        # Calculate relative path
                        rel_path = log_file.relative_to(STORAGE_DIR)
                        dest_file = backup_dest / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(log_file, dest_file)
                        size = log_file.stat().st_size
                        total_bytes += size
                        backed_up.append((log_file, size))

    print("=" * 64)
    print("             DCASS LOG BACKUP & ROTATION")
    print("=" * 64)
    print(f"[+] Backup Archive Created:")
    print(f"    {backup_dest}")
    print(f"[+] Total Files Archived: {len(backed_up)} ({total_bytes / 1024:.1f} KB)")

    if backup_only:
        print("[*] Backup-only mode: Logs preserved without truncation.")
        print("=" * 64)
        return

    # Truncate active logs and remove obsolete sub-logs
    truncated = 0
    removed_historical = 0

    for log_file, _ in backed_up:
        if log_file.name in ACTIVE_LOG_NAMES:
            # Truncate in-place so daemon process file descriptors stay open
            try:
                with open(log_file, "w") as f:
                    f.truncate(0)
                truncated += 1
            except Exception as e:
                print(f"[!] Warning truncating {log_file.name}: {e}")
        else:
            # Historical timestamped log: can safely remove
            try:
                log_file.unlink()
                removed_historical += 1
            except Exception as e:
                print(f"[!] Warning removing {log_file.name}: {e}")

    print(f"[+] Active Service Logs Truncated to 0B: {truncated}")
    if removed_historical > 0:
        print(f"[+] Historical Old Logs Removed:        {removed_historical}")
    print("[+] All active logs are now fresh and clean.")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="DCASS Log Backup & Clean Utility")
    parser.add_argument("--backup-only", action="store_true", help="Archive without resetting active log files")
    args = parser.parse_args()

    backup_and_clean_logs(backup_only=args.backup_only)


if __name__ == "__main__":
    main()
