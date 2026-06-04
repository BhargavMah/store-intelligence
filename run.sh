#!/usr/bin/env bash
# run.sh — One command to process all CCTV clips and feed events into the API
# Usage: bash run.sh [STORE_DIR] [STORE_ID] [CLIP_START] [API_URL]
#
# Example:
#   bash run.sh clips/Store1/ STORE_BLR_001 "2026-04-10T12:00:00" http://localhost:8000

set -euo pipefail

STORE_DIR="${1:-clips/Store1/}"
STORE_ID="${2:-STORE_BLR_002}"
CLIP_START="${3:-2026-04-10T12:00:00}"
API_URL="${4:-}"
OUT_FILE="data/events_${STORE_ID}.jsonl"

echo "=== Purplle Store Intelligence — Detection Pipeline ==="
echo "Store:      $STORE_ID"
echo "Clips dir:  $STORE_DIR"
echo "Clip start: $CLIP_START"
echo "Output:     $OUT_FILE"
echo ""

mkdir -p data

# Run detection on all clips in store directory
python detect.py \
  --store-dir "$STORE_DIR" \
  --store     "$STORE_ID" \
  --start     "$CLIP_START" \
  --out       "$OUT_FILE"

echo ""
echo "=== Detection complete. Events written to $OUT_FILE ==="
echo "Event count: $(wc -l < "$OUT_FILE")"
echo ""

# If API URL is provided, ingest events in batches of 100
if [ -n "$API_URL" ]; then
  echo "=== Ingesting events to API: $API_URL ==="

  # Export variables so they are visible inside the Python script
  export OUT_FILE API_URL

  python3 - <<'PYEOF'
import json
import sys
import os
import urllib.request
import urllib.error

out_file = os.environ.get("OUT_FILE", "data/events.jsonl")
api_url  = os.environ.get("API_URL", "http://localhost:8000")
batch_size = 100

events = []
with open(out_file) as f:
    for line in f:
        line = line.strip()
        if line:
            events.append(json.loads(line))

total_accepted = 0
total_rejected = 0

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
            print(f"  Batch {i//batch_size + 1}: accepted={d['accepted']}, rejected={d['rejected']}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"  Batch {i//batch_size + 1}: HTTP {e.code} - {body}", file=sys.stderr)

print(f"\nTotal: accepted={total_accepted}, rejected={total_rejected}")
PYEOF

fi

echo ""
echo "=== Pipeline complete ==="
