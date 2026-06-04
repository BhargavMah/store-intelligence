# Store Intelligence — Purplle Retail Analytics

> **End-to-end CCTV → Live Store Analytics pipeline.**  
> YOLOv8 detection · ByteTrack tracking · Re-ID · FastAPI · PostgreSQL · Live Dashboard

---

## Setup in 5 Commands

```bash
# 1. Clone the repository
git clone <your-repo-url> store-intelligence && cd store-intelligence

# 2. Start the API + Database + Dashboard (Docker)
docker compose up -d

# 3. Verify the API is healthy
curl http://localhost:8000/health

# 4. Open the live dashboard
# → http://localhost:3000   (served by nginx in Docker)
# → or open dashboard.html directly in your browser (points to localhost:8000)

# 5. Load POS data and check metrics
curl -X POST http://localhost:8000/admin/load-pos \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "/data/pos_transactions.csv", "store_id": "ST1008"}'

curl http://localhost:8000/stores/ST1008/metrics
```

---

## Project Structure

```
store-intelligence/
├── detect.py             # Main YOLOv8 + ByteTrack detection script
├── tracker.py            # Re-ID, session state machine, event emission logic
├── emit.py               # Pydantic schema validation + JSONL writer
├── staff_filter.py       # Colour histogram + frequency-based staff classifier
├── zone_mapper.py        # (camera, centroid) → zone_id polygon mapping
├── schema.py             # Shared Pydantic v2 event schema
├── store_layout.json     # Zone polygon definitions for all stores
├── run.sh                # One-command: process clips → ingest to API
├── simulate_live_feed.py # Sends synthetic events for live dashboard demo
├── dashboard.html        # Live analytics dashboard (Part E bonus)
├── Dockerfile            # API container (Python 3.11-slim)
├── docker-compose.yml    # PostgreSQL + API + Dashboard nginx services
├── nginx.conf            # Nginx config for dashboard service
├── requirements.api.txt  # API dependencies
├── requirements.detect.txt  # Detection pipeline dependencies
└── app/
    ├── main.py           # FastAPI entrypoint, routes, error handlers
    ├── database.py       # SQLAlchemy ORM (SQLite dev / PostgreSQL prod)
    ├── ingestion.py      # POST /events/ingest (idempotent, batch, partial success)
    ├── metrics.py        # GET /stores/{id}/metrics
    ├── funnel.py         # GET /stores/{id}/funnel (session-deduped)
    ├── heatmap.py        # GET /stores/{id}/heatmap (normalised 0–100)
    ├── anomalies.py      # GET /stores/{id}/anomalies (queue spike, conversion drop, dead zone)
    ├── health.py         # GET /health (STALE_FEED detection)
    ├── pos_loader.py     # POS CSV loader + 5-min billing window conversion correlation
    ├── middleware.py     # Structured JSON request logging (trace_id, latency_ms)
    └── tests/
        ├── test_api.py            # 25 API endpoint tests (idempotency, edge cases)
        ├── test_pipeline.py       # 26 detection pipeline unit tests
        └── test_coverage_boost.py # 16 additional tests → 94% total coverage
```

---

## Running the Detection Pipeline

### Clips already placed in the project:

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

### Prerequisites

```bash
pip install -r requirements.detect.txt
```

### Run on the provided clips (Windows-friendly)

```bash
# Use run_clips.py — handles spaces in filenames, auto-detects camera IDs

# Process Store 1 (4 cameras) → detect → ingest to API
python run_clips.py --store 1 --api http://localhost:8000

# Process Store 2 (4 cameras)
python run_clips.py --store 2 --api http://localhost:8000

# Both stores at once
python run_clips.py --store all --api http://localhost:8000

# Only detect (write to file, skip API ingestion)
python run_clips.py --store 1 --no-ingest
```

### Single clip manually

```bash
python detect.py \
  --video   "clips/Store1/CAM 3 - entry.mp4" \
  --store   STORE_BLR_001 \
  --camera  CAM_ENTRY_01 \
  --start   "2026-04-10T12:00:00" \
  --out     data/events.jsonl
```

**Camera auto-detection from filename:**
- `entry*.mp4` → `CAM_ENTRY_01`
- `billing*.mp4` or `bill*.mp4` → `CAM_BILLING_01`
- Anything else → `CAM_FLOOR_01`


```bash
bash run.sh clips/Store1/ STORE_BLR_001 "2026-04-10T12:00:00" http://localhost:8000
```

---

## Feeding Events into the API

### Run the live feed simulator (for dashboard demo)

```bash
pip install requests
python simulate_live_feed.py
# → Sends synthetic events every 1.5s while the dashboard shows them live
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events/ingest` | POST | Ingest up to 500 events. Idempotent by `event_id`. Partial success on malformed events. |
| `/stores/{id}/metrics` | GET | Real-time metrics: visitors, conversion rate, avg dwell per zone, queue depth, abandonment rate |
| `/stores/{id}/funnel` | GET | Conversion funnel: ENTRY → ZONE_VISIT → BILLING_QUEUE → PURCHASE with drop-off % |
| `/stores/{id}/heatmap` | GET | Zone visit frequency + avg dwell, normalised 0–100. `data_confidence` flag if < 20 sessions |
| `/stores/{id}/anomalies` | GET | Active anomalies: queue spike, conversion drop, dead zones, stale feed. Severity: INFO/WARN/CRITICAL |
| `/health` | GET | Service status, last event per store, STALE_FEED warning if > 10 min lag |
| `/docs` | GET | Interactive Swagger UI |

**Interactive docs:** http://localhost:8000/docs

---

## Live Dashboard (Part E — Bonus)

Open **http://localhost:3000** (Docker) or open `dashboard.html` directly.

Features:
- Real-time KPI cards (visitors, conversion, queue depth, abandonment)
- Animated conversion funnel with drop-off percentages
- Hot/cold zone heatmap (normalised scores with color coding)
- Live anomaly feed with severity-coded alerts and action suggestions
- Live event log showing incoming events as they arrive
- Auto-refreshes every 3 seconds

---

## Running Tests

```bash
# Navigate to the app directory
cd app

# Install test dependencies
pip install -r ../requirements.api.txt

# Run all tests with coverage
python -m pytest tests/ -v --cov=. --cov-report=term-missing

# Run only API tests
python -m pytest tests/test_api.py -v

# Run only pipeline tests
python -m pytest tests/test_pipeline.py -v
```

**Current coverage: 94%** (67 tests)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./store_intelligence.db` | DB connection string (switches to PostgreSQL in Docker) |
| `POS_CSV_PATH` | (none) | Path to POS CSV auto-loaded on startup |
| `POS_STORE_ID` | `STORE_BLR_002` | Store ID for POS data mapping |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model to use (`yolov8n`, `yolov8s`, `yolov8m`) |
| `USE_VLM` | (not set) | Set to `1` to enable Gemini VLM for staff classification |

---

## Edge Cases Handled

| Case | Handling |
|------|---------|
| Group entry (2–4 people) | ByteTrack assigns separate track IDs → N ENTRY events |
| Staff movement | 48-dim HSV histogram + track frequency + multi-camera presence → `is_staff=true` |
| Re-entry | Cosine similarity Re-ID (threshold 0.75) matches prior session → REENTRY event |
| Partial occlusion | Kalman filter extrapolates position; confidence reflects actual detection score |
| Billing queue abandonment | Visitor leaves billing zone without POS match → `BILLING_QUEUE_ABANDON` |
| Camera overlap deduplication | Same embedding within 10s across cameras → same `visitor_id` |
| Empty store periods | Zero tracks → zero events; API returns 0-counts, never null |
| DB unavailable | HTTP 503 with structured JSON body; no raw stack traces |
| Duplicate event ingest | `INSERT OR IGNORE` on `event_id`; safe to call twice (idempotent) |
| Zero purchases | `conversion_rate = 0.0`, not null; funnel PURCHASE stage = 0 |
| Large batches | 500-event limit enforced with HTTP 413; partial success on malformed events |

---

## Docker Services

```
docker compose up -d

Services:
  db        → PostgreSQL 15 (port 5432, volume pgdata)
  api       → FastAPI (port 8000, 2 workers)
  dashboard → nginx serving dashboard.html (port 3000)

Health checks:
  db  → pg_isready -U api -d storedb (5s interval, 12 retries)
  api → GET /health (10s interval, after 15s start delay)
```

---

## Submission Checklist

- [x] `docker compose up` starts everything (no manual steps beyond git clone)
- [x] Detection pipeline documented — `python detect.py --help`
- [x] `POST /events/ingest` accepts events without 5xx
- [x] `GET /stores/STORE_BLR_002/metrics` returns valid JSON
- [x] `DESIGN.md` — architecture + AI-Assisted Decisions (>250 words)
- [x] `CHOICES.md` — model selection, schema design, API architecture (>250 words)
- [x] Prompt blocks at top of each test file
- [x] Dashboard URL: `http://localhost:3000`
