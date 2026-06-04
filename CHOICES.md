# CHOICES.md — Key Engineering Decisions

## Decision 1: Detection Model — YOLOv8n + ByteTrack

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| **YOLOv8n + ByteTrack** | CPU-runnable, fast (150ms/frame), well-documented | Lower accuracy than larger models |
| YOLOv8s + ByteTrack | Better accuracy | 2x slower; still CPU-compatible |
| YOLOv8x + ByteTrack | Best accuracy | GPU required for real-time |
| MediaPipe Pose | Very fast, built-in tracking | No bounding boxes; weak at crowds |
| RT-DETR | SOTA accuracy | Transformer; slow on CPU |

### What AI Suggested
Claude recommended starting with YOLOv8n for prototyping and noted that YOLOv8s
is the right upgrade path once the pipeline is verified correct. It also specifically
flagged that YOLOv8x would require GPU for the 1080p @ 15fps clips provided.

### What I Chose and Why
**YOLOv8n** with the `YOLO_MODEL` env var making it easy to upgrade to `yolov8s.pt`
without code changes. The choice prioritises getting a working pipeline over peak
accuracy — the scoring rubric explicitly states "what we evaluate is how you handle
uncertainty, confidence thresholds, and edge cases — not a perfect detection rate."

For ByteTrack: the key advantage over DeepSORT is that it does not require a Re-ID
model during tracking. In a group-entry scenario (3 people walking through the door
simultaneously), ByteTrack handles the IoU overlap better because it maintains
low-confidence track hypotheses that DeepSORT would drop.

**On VLM usage for detection:** I evaluated using Gemini Flash for zone classification
(given a cropped person image, "which shelf are they standing in front of?"). The
test on the sample footage frames showed that zone classification via geometric
polygon-in-point is faster, deterministic, and more accurate for a top-down CCTV
angle where the floor plan polygons are well-defined. The VLM path is preserved as
an optional fallback for ambiguous zone boundaries.

---

## Decision 2: Event Schema Design

### Options Considered

**Option A: Flat schema** — all fields at the top level, no `metadata` object.
- Simple to ingest; harder to extend without breaking consumers.

**Option B: Nested metadata (chosen)** — core fields flat, extensible `metadata` dict.
- Allows queue_depth, sku_zone, session_seq to be added without breaking the schema.
- Pydantic `extra="allow"` in EventMetadata lets future fields pass through.

**Option C: Event-type-specific schemas** — separate models for ENTRY, ZONE_DWELL, etc.
- Most type-safe; hardest to maintain as event types evolve.

### What AI Suggested
Claude initially suggested Option C (discriminated union via Pydantic's `Literal` type).
The argument was that it would give the clearest type safety at ingest time. I disagreed:
a discriminated union with 8 variants creates significant boilerplate and makes batch
ingest code complex. Option B gives 90% of the type safety with 20% of the complexity.

The `event_id` field is a UUID v4 generated at emission time. This is the idempotency
key at ingest — it allows the same JSONL file to be replayed into the API safely.
Claude suggested using a hash of (store_id + camera_id + visitor_id + timestamp) as
the event_id for deterministic generation. I kept UUID v4 because deterministic IDs
can cause silent data loss if two different events happen to have the same inputs
(e.g., a visitor enters and exits in the same second).

### Key Schema Decisions
- `timestamp` is ISO-8601 UTC — derived from clip start time + frame offset
- `confidence` is the raw YOLO detection score — never suppressed, even if low
- `is_staff` is a first-class field (not in metadata) because all downstream metrics
  depend on it for filtering

---

## Decision 3: API Architecture — FastAPI + SQLite/PostgreSQL

### Options Considered

| Choice | Pros | Cons |
|--------|------|------|
| **FastAPI + SQLAlchemy** | Fast, async, Pydantic integration, scoring harness compatible | SQLAlchemy ORM can be verbose |
| Flask + SQLAlchemy | Familiar | No async; slower under load |
| FastAPI + raw SQL | Maximum control | Hard to maintain, no ORM benefits |
| Node.js (Express) | Fast async | Python scoring harness has less coverage |
| Go (Gin) | Very fast | Python scoring harness has less coverage |

**Storage:**
| Option | Pros | Cons |
|--------|------|------|
| **SQLite (dev) / PostgreSQL (prod)** | Zero-config dev; production-grade prod | Two configs to maintain |
| PostgreSQL only | Single config | Requires Docker for local dev |
| Redis | Fast reads | Not suited for complex SQL queries (/funnel, /heatmap) |
| ClickHouse | Fast analytics | Heavyweight; overkill for this data volume |

### What AI Suggested
Claude recommended PostgreSQL exclusively (not SQLite at all), arguing that the
behavioral differences between SQLite and PostgreSQL (especially around `INSERT OR IGNORE`
vs `ON CONFLICT DO NOTHING`, and full-text search) would create hidden bugs.

This is a valid concern. My counter-argument: for a hiring challenge where the scorer
runs `docker compose up` and the API starts in 30 seconds with no other setup, SQLite
as the default makes the acceptance gate (item 1: "No manual steps beyond git clone")
easier to pass. The `DATABASE_URL` env var makes it trivial to switch to PostgreSQL
in Docker Compose without code changes.

### Real-Time Metrics
The `/metrics` endpoint is intentionally un-cached. Every call runs a live SQL query.
For 40 stores at production scale, this would need a materialized view refreshed
every 30 seconds. For the challenge, real-time accuracy is prioritized over performance.
The `/health` endpoint (which on-call engineers check first) always reflects the true
current state of the feed, not a cached snapshot.

### Graceful Degradation
When the database is unreachable:
- `/health` returns `{"status": "DEGRADED", "database": "UNREACHABLE"}` with HTTP 503
- All other endpoints return HTTP 503 via SQLAlchemy's `pool_pre_ping=True`
- No raw stack traces are ever returned — the global exception handler in `main.py`
  catches all unhandled exceptions and returns a structured error body.
