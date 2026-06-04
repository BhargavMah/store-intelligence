"""
run_clips.py  — Windows-friendly script to process the provided CCTV clips
                and feed all events into the running API.

Usage:
    # Process Store 1 (4 cameras)
    python run_clips.py --store 1

    # Process Store 2 (4 cameras)
    python run_clips.py --store 2

    # Process both stores
    python run_clips.py --store all

    # Process a single clip manually
    python run_clips.py --video "clips/Store1/CAM 3 - entry.mp4" --store-id STORE_BLR_001

    # Override the API URL (default: http://localhost:8000)
    python run_clips.py --store 1 --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Map clip filenames → camera IDs ─────────────────────────────────────────

def camera_id_from_name(filename: str) -> str:
    name = Path(filename).stem.lower()
    if "entry" in name:
        return "CAM_ENTRY_01"
    if "billing" in name or "bill" in name:
        return "CAM_BILLING_01"
    return "CAM_FLOOR_01"


# ── Store config ─────────────────────────────────────────────────────────────

STORE_CONFIG = {
    "1": {
        "store_id": "STORE_BLR_001",
        "clips_dir": "clips/Store1",
        "clip_start": "2026-04-10T12:00:00",
        "out_file": "data/events_STORE_BLR_001.jsonl",
    },
    "2": {
        "store_id": "STORE_BLR_002",
        "clips_dir": "clips/Store2",
        "clip_start": "2026-04-10T12:00:00",
        "out_file": "data/events_STORE_BLR_002.jsonl",
    },
}


# ── Ingest events file to API ─────────────────────────────────────────────────

def ingest_file(jsonl_path: str, api_url: str, batch_size: int = 100) -> None:
    events = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        print(f"  [ingest] No events in {jsonl_path}")
        return

    print(f"  [ingest] Sending {len(events)} events in batches of {batch_size}…")
    total_accepted = total_rejected = 0

    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        payload = json.dumps({"events": batch}).encode()
        req = urllib.request.Request(
            f"{api_url}/events/ingest",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read())
                total_accepted += d.get("accepted", 0)
                total_rejected += d.get("rejected", 0)
                print(f"    Batch {i // batch_size + 1}: ✓ accepted={d['accepted']}  ✗ rejected={d['rejected']}")
        except urllib.error.HTTPError as e:
            print(f"    Batch {i // batch_size + 1}: HTTP {e.code} — {e.read().decode()[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"    Batch {i // batch_size + 1}: ERROR — {e}", file=sys.stderr)

    print(f"  [ingest] Done: {total_accepted} accepted, {total_rejected} rejected")


# ── Process one store ─────────────────────────────────────────────────────────

def process_store(config: dict, api_url: str) -> None:
    store_id    = config["store_id"]
    clips_dir   = config["clips_dir"]
    clip_start  = config["clip_start"]
    out_file    = config["out_file"]

    print(f"\n{'='*60}")
    print(f"  Store:      {store_id}")
    print(f"  Clips dir:  {clips_dir}")
    print(f"  Clip start: {clip_start}")
    print(f"  Output:     {out_file}")
    print(f"{'='*60}")

    clips_path = Path(clips_dir)
    if not clips_path.exists():
        print(f"  [ERROR] Clips directory not found: {clips_dir}", file=sys.stderr)
        return

    # Find all video clips
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    clip_files = [f for f in sorted(clips_path.iterdir())
                  if f.suffix.lower() in video_extensions]

    if not clip_files:
        print(f"  [ERROR] No video clips found in {clips_dir}", file=sys.stderr)
        return

    print(f"\n  Found {len(clip_files)} clip(s):")
    for cf in clip_files:
        cam_id = camera_id_from_name(cf.name)
        print(f"    {cf.name}  →  {cam_id}")

    # Run detection
    os.makedirs("data", exist_ok=True)
    cmd = [
        sys.executable, "detect.py",
        "--store-dir", str(clips_dir),
        "--store",     store_id,
        "--start",     clip_start,
        "--out",       out_file,
    ]

    print(f"\n  [detect] Running: {' '.join(cmd)}\n")
    ret = os.system(" ".join(f'"{c}"' if " " in c else c for c in cmd))

    if ret != 0:
        print(f"  [ERROR] Detection failed (exit code {ret})", file=sys.stderr)
        return

    # Ingest to API
    if api_url and Path(out_file).exists():
        print(f"\n  [ingest] Ingesting {out_file} → {api_url}")
        ingest_file(out_file, api_url)
    elif not Path(out_file).exists():
        print(f"  [WARN] Output file not found: {out_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Run Store Intelligence detection pipeline on the provided CCTV clips"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--store", choices=["1", "2", "all"],
                       help="Which store to process (1, 2, or all)")
    group.add_argument("--video", help="Path to a single video clip")

    p.add_argument("--store-id", default="STORE_BLR_001",
                   help="Store ID when using --video (default: STORE_BLR_001)")
    p.add_argument("--camera", default=None,
                   help="Camera ID when using --video (auto-detected if not set)")
    p.add_argument("--start", default="2026-04-10T12:00:00",
                   help="Clip start ISO timestamp (default: 2026-04-10T12:00:00)")
    p.add_argument("--out", default=None,
                   help="Output JSONL path (default: data/events_<store_id>.jsonl)")
    p.add_argument("--api", default="http://localhost:8000",
                   help="API URL to ingest events into (default: http://localhost:8000)")
    p.add_argument("--no-ingest", action="store_true",
                   help="Only run detection, skip ingesting to API")

    args = p.parse_args()
    api_url = None if args.no_ingest else args.api

    if args.store:
        stores = ["1", "2"] if args.store == "all" else [args.store]
        for s in stores:
            process_store(STORE_CONFIG[s], api_url)

    elif args.video:
        # Single clip mode
        video = args.video
        cam_id = args.camera or camera_id_from_name(video)
        out = args.out or f"data/events_{args.store_id}.jsonl"
        os.makedirs("data", exist_ok=True)

        cmd = [
            sys.executable, "detect.py",
            "--video",  video,
            "--camera", cam_id,
            "--store",  args.store_id,
            "--start",  args.start,
            "--out",    out,
        ]
        print(f"Running: {' '.join(cmd)}")
        ret = os.system(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        if ret == 0 and api_url and Path(out).exists():
            ingest_file(out, api_url)

    print("\nDone! Check the dashboard at http://localhost:3000")


if __name__ == "__main__":
    main()
