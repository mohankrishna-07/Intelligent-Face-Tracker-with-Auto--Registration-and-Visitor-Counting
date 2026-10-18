"""CLI reporting utility for querying SQLite visitor statistics and verifying against events.log."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import AppConfig
from app.db import DatabaseManager


def generate_report(config_path: str = "config.json") -> None:
    """Generate and display visitor counts, entry/exit audit logs, and log file cross-check."""
    cfg = AppConfig.load(config_path)
    db = DatabaseManager(cfg.storage, base_dir=cfg.base_dir)

    print("\n" + "=" * 70)
    print("      INTELLIGENT FACE TRACKER - VISITOR ANALYTICS REPORT")
    print("=" * 70)

    # 1. Total Unique Visitors
    unique_count = db.get_unique_visitor_count()
    print(f"\n[SUMMARY METRICS]")
    print(f"  • Total Unique Visitors Registered: {unique_count}")

    # 2. Per-Face Registry Details
    identities = db.load_all_identities()
    print(f"\n[REGISTERED FACES GALLERY] ({len(identities)} Total)")
    print(f"  {'Face ID':<12} | {'First Seen':<24} | {'Last Seen':<24} | {'Embeddings'}")
    print("  " + "-" * 66)
    for fid, first_seen, last_seen, _, count in identities:
        print(f"  {fid:<12} | {first_seen[:19]:<24} | {last_seen[:19]:<24} | {count} samples")

    # 3. Entry / Exit Event History
    events = db.get_event_history(limit=50)
    print(f"\n[RECENT ENTRY / EXIT AUDIT TRAIL] (Showing last {len(events)} events)")
    print(f"  {'Type':<7} | {'Face ID':<11} | {'Track':<6} | {'Timestamp':<20} | {'Confidence':<10} | {'Crop Path'}")
    print("  " + "-" * 85)
    for ev in reversed(events):
        ev_type = ev["event_type"].upper()
        fid = ev["face_id"]
        tid = ev["track_id"]
        ts = ev["timestamp"][:19]
        conf = f"{ev['confidence']:.2f}"
        img = ev["image_path"]
        print(f"  {ev_type:<7} | {fid:<11} | {tid:<6} | {ts:<20} | {conf:<10} | {img}")

    # 4. Cross-Verification with events.log
    print("\n[LOG FILE CROSS-VERIFICATION]")
    log_file = (cfg.base_dir / cfg.storage.log_file).resolve()
    if not log_file.exists():
        print(f"  [WARN] Log file not found at {log_file}")
    else:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()

        log_entries = len(re.findall(r'\[STATE ENTRY\]', log_content))
        log_exits = len(re.findall(r'\[STATE EXIT\]|\[FLUSH EXIT\]', log_content))
        log_registered = len(re.findall(r'\[REGISTERED\]', log_content))
        log_recognized = len(re.findall(r'\[RECOGNIZED\]', log_content))

        # Query DB event counts
        db_entries = sum(1 for e in events if e["event_type"] == "entry")
        db_exits = sum(1 for e in events if e["event_type"] == "exit")

        print(f"  - Log File Location:        {log_file}")
        print(f"  - Logged Auto-Registrations: {log_registered}")
        print(f"  - Logged Re-Identifications: {log_recognized}")
        print(f"  - Logged ENTRY events:       {log_entries}")
        print(f"  - Logged EXIT events:        {log_exits}")
        print(f"  - DB Unique Faces Count:     {unique_count}")

        # Verification check
        if log_registered == unique_count:
            print(f"  [VERIFIED] DB faces count ({unique_count}) matches logged auto-registrations exactly!")
        else:
            print(f"  [INFO] DB has {unique_count} faces, log file records {log_registered} registrations.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visitor report from SQLite database.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    args = parser.parse_args()
    generate_report(args.config)
