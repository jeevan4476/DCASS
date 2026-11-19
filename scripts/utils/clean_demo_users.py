#!/usr/bin/env python3
"""
DCASS User Database Cleanup Utility.
Removes test/ephemeral accounts from storage/auth.db while preserving official demo users.
"""

import sqlite3
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.auth import hash_password, init_db

DB_PATH = PROJECT_ROOT / "storage" / "auth.db"
SHARED_CHANNEL_DIR = PROJECT_ROOT / "storage" / "shared_channel"

CORE_USERS = {"alice", "bob", "charlie"}
DEFAULT_PASSWORD = "password123"


def clean_users(keep_usernames=CORE_USERS, reset_passwords=True):
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT id, username FROM users")
    all_users = cursor.fetchall()
    total_before = len(all_users)

    deleted_count = 0
    existing_map = {}

    for uid, uname in all_users:
        if uname in keep_usernames:
            existing_map[uname] = uid
        else:
            cursor.execute("DELETE FROM users WHERE id = ?", (uid,))
            deleted_count += 1

    conn.commit()

    # Ensure all core users exist and have standard passwords
    hashed = hash_password(DEFAULT_PASSWORD)
    for uname in sorted(keep_usernames):
        if uname in existing_map:
            if reset_passwords:
                cursor.execute(
                    "UPDATE users SET password_hash = ? WHERE username = ?",
                    (hashed, uname),
                )
        else:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (uname, hashed),
            )

    conn.commit()

    # Re-fetch active IDs
    cursor.execute("SELECT id, username FROM users")
    active_users = cursor.fetchall()
    active_ids = {str(uid) for uid, _ in active_users}

    # Also clean orphaned shared_channel user directories and sent messages
    cleaned_dirs = 0
    if SHARED_CHANNEL_DIR.exists():
        for d in SHARED_CHANNEL_DIR.iterdir():
            if d.is_dir() and d.name.isdigit():
                if d.name not in active_ids:
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)
                    cleaned_dirs += 1
                else:
                    # Clean out sent messages to non-active users
                    for sf in d.glob("_sent_*.json"):
                        try:
                            import json
                            data = json.loads(sf.read_text())
                            if data.get("recipient_username") not in keep_usernames:
                                sf.unlink(missing_ok=True)
                        except Exception:
                            pass

    conn.close()

    print("=" * 60)
    print("           DCASS AUTHENTICATION DATABASE CLEANUP")
    print("=" * 60)
    print(f"[+] Total users before cleanup: {total_before}")
    print(f"[+] Deleted test users:        {deleted_count}")
    if cleaned_dirs > 0:
        print(f"[+] Cleaned orphaned inboxes:  {cleaned_dirs}")
    print(f"\nOfficial Demo Accounts (Password: {DEFAULT_PASSWORD}):")
    for uid, uname in active_users:
        print(f"  - Username: {uname:<10} (ID: {uid})")
    print("=" * 60)


if __name__ == "__main__":
    clean_users()
