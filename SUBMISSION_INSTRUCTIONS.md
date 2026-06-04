# Store Intelligence — Submission Instructions

## Instructions to Run

### Requirements
- Docker Desktop (or Docker Engine + Docker Compose v2)
- Git

### Step 1 — Clone and Start

```bash
git clone <your-repo-url> store-intelligence
cd store-intelligence
docker compose up -d
```

This starts **3 services**:
| Service | URL | Description |
|---------|-----|-------------|
| `db` | (internal, port 5432) | PostgreSQL 15 database |
| `api` | http://localhost:8000 | FastAPI Store Intelligence API |
| `dashboard` | http://localhost:3000 | Live analytics dashboard |

### Step 2 — Wait for healthy status

```bash
docker compose ps
# All services should show "healthy" or "running"
```

### Step 3 — Verify the API

```bash
curl http://localhost:8000/health
# Expected: {"status":"OK","database":"OK","stores":{}}

curl http://localhost:8000/stores/STORE_BLR_002/metrics
# Expected: {"store_id":"STORE_BLR_002","unique_visitors":0,...}
```

### Step 4 — Load POS data (included in repo)

```bash
curl -X POST http://localhost:8000/admin/load-pos \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "/data/pos_transactions.csv", "store_id": "ST1008"}'
```

### Step 5 — Ingest sample events

```bash
# Option A: Use the included sample_events.jsonl
python - <<'EOF'
import json, urllib.request
events = [json.loads(l) for l in open("data/sample_events.jsonl") if l.strip()]
data = json.dumps({"events": events}).encode()
req = urllib.request.Request("http://localhost:8000/events/ingest", data=data,
      headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req) as r:
    print("Result:", json.loads(r.read()))
EOF
```

### Step 6 — Run the live dashboard

Open http://localhost:3000 in your browser.

To see it with live data, run the simulator in a separate terminal:
```bash
pip install requests
python simulate_live_feed.py
```

---

## Running the Detection Pipeline Against CCTV Clips

The clips from the provided zip files are placed in:
```
clips/
├── Store1/
│   ├── CAM 1 - zone.mp4        → CAM_FLOOR_01
│   ├── CAM 2 - zone.mp4        → CAM_FLOOR_01
│   ├── CAM 3 - entry.mp4       → CAM_ENTRY_01
│   └── CAM 5 - billing.mp4     → CAM_BILLING_01
└── Store2/
    ├── billing_area.mp4         → CAM_BILLING_01
    ├── entry 1.mp4              → CAM_ENTRY_01
    ├── entry 2.mp4              → CAM_ENTRY_01
    └── zone.mp4                 → CAM_FLOOR_01
```

Camera IDs are **auto-detected from the filename** (`entry*` → `CAM_ENTRY_01`, `billing*` → `CAM_BILLING_01`, everything else → `CAM_FLOOR_01`).

### Install detection dependencies

```bash
pip install -r requirements.detect.txt
# (ultralytics, opencv-python, numpy, pydantic — already installed if you ran setup)
```

### Option A — Easy: use run_clips.py (Windows-friendly, handles spaces in filenames)

```bash
# Make sure the API is running first (docker compose up -d OR python -m uvicorn main:app)

# Process Store 1 (all 4 cameras → ingest to API)
python run_clips.py --store 1 --api http://localhost:8000

# Process Store 2 (all 4 cameras → ingest to API)
python run_clips.py --store 2 --api http://localhost:8000

# Process both stores
python run_clips.py --store all --api http://localhost:8000

# Just detect, don't ingest yet (write events to file only)
python run_clips.py --store 1 --no-ingest
```

### Option B — Manual: run detect.py directly

```bash
# Store 1 — all clips → events file → ingest
python detect.py --store-dir "clips/Store1/" --store STORE_BLR_001 --start "2026-04-10T12:00:00" --out data/events_STORE_BLR_001.jsonl
python run_clips.py --store 1 --api http://localhost:8000  # ingest from file

# Store 2 — all clips
python detect.py --store-dir "clips/Store2/" --store STORE_BLR_002 --start "2026-04-10T12:00:00" --out data/events_STORE_BLR_002.jsonl

# Single clip (e.g. just the entry camera of Store 1)
python detect.py --video "clips/Store1/CAM 3 - entry.mp4" --camera CAM_ENTRY_01 --store STORE_BLR_001 --start "2026-04-10T12:00:00" --out data/events_entry.jsonl
```

### After detection — verify results

```bash
# Check how many events were generated
# (Windows PowerShell)
(Get-Content data/events_STORE_BLR_001.jsonl).Count

# Ingest the file manually if needed
python -c "
import json, urllib.request
events = [json.loads(l) for l in open('data/events_STORE_BLR_001.jsonl') if l.strip()]
print(f'Ingesting {len(events)} events...')
data = json.dumps({'events': events[:500]}).encode()
req = urllib.request.Request('http://localhost:8000/events/ingest', data=data, headers={'Content-Type':'application/json'}, method='POST')
with urllib.request.urlopen(req) as r: print(json.loads(r.read()))
"

# Check metrics after ingestion
# (Windows PowerShell)
Invoke-WebRequest http://localhost:8000/stores/STORE_BLR_001/metrics | Select-Object -ExpandProperty Content
```

---

## Running Tests

```bash
cd app
pip install -r ../requirements.api.txt
python -m pytest tests/ -v --cov=. --cov-report=term-missing
# Expected: 67 passed, 94% coverage
```

---

## API Interactive Docs

Open http://localhost:8000/docs (Swagger UI) to explore all endpoints interactively.

---

## Environment Variables (all optional — defaults work out of the box)

| Variable | Docker default | Description |
|----------|---------------|-------------|
| `DATABASE_URL` | `postgresql://api:secret@db/storedb` | DB connection |
| `POS_CSV_PATH` | `/data/pos_transactions.csv` | Auto-loaded POS file |
| `POS_STORE_ID` | `ST1008` | Store ID for POS mapping |
| `YOLO_MODEL` | `yolov8n.pt` | Detection model (n/s/m) |

---

## Troubleshooting

**API not responding?**
```bash
docker compose logs api
```

**Database not ready?**
```bash
docker compose logs db
# Wait for: "database system is ready to accept connections"
```

**Port conflicts?**
```bash
# Change ports in docker-compose.yml if 8000 or 3000 are in use
```
