# DESIGN.md — Store Intelligence System Architecture

## Overview

This system builds a complete retail analytics pipeline from raw CCTV footage to a live
intelligence API. The architecture is designed around one north-star metric: **offline
store conversion rate** — the fraction of unique visitors who completed a purchase.

---

## System Architecture

```
Raw CCTV Clips (1080p, 15fps)
        │
        ▼
┌─────────────────────────┐
│   detect.py             │  YOLOv8n (person detection)
│   + ByteTrack           │  ByteTrack (multi-object tracking)
│   + Colour Re-ID        │  48-dim HSV histogram embedding
└──────────┬──────────────┘
           │ structured events (JSONL)
           ▼
┌─────────────────────────┐
│   schema.py             │  Pydantic v2 validation
│   emit.py               │  JSONL writer
└──────────┬──────────────┘
           │ POST /events/ingest
           ▼
┌─────────────────────────┐
│   FastAPI API           │  Python 3.11
│   SQLAlchemy ORM        │  PostgreSQL (prod) / SQLite (dev)
│   Structured logging    │  JSON per request, trace_id header
└──────────┬──────────────┘
           │
    ┌──────┴───────┐
    ▼              ▼
/metrics       /funnel
/heatmap       /anomalies
/health
```

---

## Detection Layer Design

### Why YOLOv8 + ByteTrack?

YOLOv8 offers the best accuracy/speed tradeoff for CPU inference on 1080p footage.
At 5 effective fps (every 3rd frame at 15fps), YOLOv8n processes each frame in ~150ms
on a modern CPU, giving real-time playback.

ByteTrack is preferred over DeepSORT because it does not require a Re-ID model during
tracking — it uses IoU-based association for the primary tracking signal and only falls
back to low-confidence associations for occluded tracks. This means it handles the
occlusion and group-entry edge cases better out of the box.

### Re-ID Strategy

A dedicated Re-ID model (OSNet) would give the best cross-camera matching accuracy,
but requires a GPU for reasonable inference speed. The implemented fallback is a
48-dimensional HSV colour histogram of the torso region, which:
- Runs in <1ms per track (pure NumPy)
- Works on CPU
- Achieves ~80% re-identification accuracy for distinctly-dressed individuals

Cross-camera matching uses cosine similarity with a threshold of 0.75. The threshold
was chosen empirically: below 0.75, false positives (matching different people) become
problematic; above 0.75, too many re-entries are missed.

### Staff Classification

The primary method is torso colour histogram matching. In practice, retail staff often
wear uniform-coloured polo shirts or aprons. Two additional signals boost confidence:
1. Track frequency (staff are seen many more times than customers)
2. Multi-camera presence (staff appear on all three camera feeds)

An optional VLM path (Gemini Flash) is available for ambiguous cases via `USE_VLM=1`.

---

## API Design

### Idempotency

`POST /events/ingest` uses `event_id` as the deduplication key. If the same event_id
is seen twice, the second occurrence is counted as "accepted" but not re-inserted.
This ensures that the detection pipeline can safely retry failed ingest batches without
data corruption.

### Conversion Rate Correlation

Since POS data has no `customer_id`, conversion is determined probabilistically:
a visitor is "converted" if they were in the billing zone within 5 minutes before
a POS transaction for the same store. This 5-minute window is configurable and
aligns with typical checkout times in a retail context.

### Session Deduplication

The funnel treats `visitor_id` as the session unit. A visitor who re-enters the store
(REENTRY event) is not double-counted — they are the same session. This is enforced
by using `DISTINCT visitor_id` in all funnel queries.

---

## AI-Assisted Decisions

### 1. ByteTrack vs DeepSORT for tracking

I asked Claude to compare ByteTrack and DeepSORT for a retail CCTV scenario with
partial occlusion. Claude correctly identified that DeepSORT's dependency on a
pre-trained Re-ID model creates a chicken-and-egg problem for CPU deployment — the
Re-ID model itself is slow on CPU. ByteTrack's IoU-only association is faster and
more robust for dense scenes. I agreed with this assessment and chose ByteTrack.

### 2. Cosine similarity threshold for Re-ID

I prompted Claude to suggest a starting threshold for HSV histogram cosine similarity
in a retail context. Claude suggested 0.80, reasoning that retail environments have
variable lighting that would make histograms drift. After testing on the sample events
(where the same visitor appears 3 times within 2 minutes), I found 0.75 gave better
recall without too many false positives. I overrode Claude's suggestion.

### 3. Probabilistic POS correlation

I asked Claude whether to (a) use ML-based demand forecasting to infer purchases from
dwell time, or (b) use the simpler time-window correlation. Claude's first suggestion
was option (a) — but this required training data we don't have yet. I chose option (b)
as more transparent, auditable, and appropriate for the data we actually have (timestamped
POS transactions). Claude agreed this was the right tradeoff for a first deployment.

---

## Trade-offs Accepted

| Decision | Trade-off |
|----------|-----------|
| YOLOv8n (nano) over larger models | Speed over accuracy; can swap to yolov8s via env var |
| HSV histogram Re-ID over OSNet | CPU-compatible; lower cross-camera accuracy |
| SQLite default over PostgreSQL | Zero-config dev; Docker Compose switches to PG |
| Frame sampling (every 3rd frame) | Misses fast movements; adequate for 15fps retail footage |
| Probabilistic POS correlation | Some conversion assignments will be wrong; no alternative without customer IDs |
