"""
main.py — FastAPI application entrypoint.

Mounts all API routes and configures middleware, error handlers, and startup tasks.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# ── Path setup (find schema.py in parent dir) ────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database import SessionLocal, create_tables, get_db
from ingestion import ingest_events
from metrics import compute_metrics
from funnel import compute_funnel
from heatmap import compute_heatmap
from anomalies import detect_anomalies
from health import compute_health
from middleware import StructuredLoggingMiddleware
from pos_loader import load_pos_csv


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run on startup: create tables and optionally seed POS data."""
    create_tables()

    # Auto-load POS CSV if path is configured
    pos_csv = os.environ.get("POS_CSV_PATH")
    pos_store = os.environ.get("POS_STORE_ID", "STORE_BLR_002")
    if pos_csv and os.path.exists(pos_csv):
        db = SessionLocal()
        try:
            n = load_pos_csv(pos_csv, db, pos_store)
            print(f"[startup] Loaded {n} POS transactions from {pos_csv}")
        finally:
            db.close()

    print("[startup] Store Intelligence API ready.")
    yield
    print("[shutdown] Store Intelligence API stopped.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Store Intelligence API",
    description=(
        "Real-time retail analytics API. "
        "Ingests CCTV-derived visitor events and exposes store metrics, "
        "conversion funnels, zone heatmaps, and operational anomalies."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(StructuredLoggingMiddleware)


# ── Global error handlers ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured error body; never expose raw stack traces."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please contact support.",
            "path": str(request.url.path),
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/events/ingest",
    summary="Ingest a batch of detection events",
    response_description="Counts of accepted and rejected events",
)
async def post_ingest(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Accept up to 500 events in a single batch.
    Idempotent: sending the same event_id twice is safe.
    Returns partial success on malformed events.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request body must be valid JSON.",
        )

    # Support both {"events": [...]} and bare [...]
    if isinstance(body, list):
        events_list = body
    elif isinstance(body, dict):
        events_list = body.get("events", [body])
    else:
        raise HTTPException(status_code=422, detail="Expected JSON array or object with 'events' key.")

    if len(events_list) > 500:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {len(events_list)} exceeds limit of 500.",
        )

    # Attach event_count to request state for logging middleware
    request.state.event_count = len(events_list)

    result = ingest_events(events_list, db)
    status_code = 200 if result["rejected"] == 0 else 207  # 207 Multi-Status
    return JSONResponse(content=result, status_code=status_code)


@app.get(
    "/stores/{store_id}/metrics",
    summary="Real-time store metrics",
)
async def get_metrics(
    store_id: str,
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Returns today's real-time metrics for the given store.
    Excludes staff events. Never cached.
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d") if date else None
        return compute_metrics(store_id, db, dt)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD format.")


@app.get(
    "/stores/{store_id}/funnel",
    summary="Conversion funnel with drop-off percentages",
)
async def get_funnel(
    store_id: str,
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Returns the conversion funnel for the given store.
    Session is the unit — re-entries do not double-count a visitor.
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d") if date else None
        return compute_funnel(store_id, db, dt)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD format.")


@app.get(
    "/stores/{store_id}/heatmap",
    summary="Zone visit frequency heatmap",
)
async def get_heatmap(
    store_id: str,
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Returns zone visit frequency and avg dwell, normalised 0–100.
    Flags data_confidence=LOW if fewer than 20 sessions.
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d") if date else None
        return compute_heatmap(store_id, db, dt)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD format.")


@app.get(
    "/stores/{store_id}/anomalies",
    summary="Active operational anomalies",
)
async def get_anomalies(
    store_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns active anomalies: queue spikes, conversion drops, dead zones, stale feeds.
    Severity levels: INFO / WARN / CRITICAL.
    """
    return detect_anomalies(store_id, db)


@app.get(
    "/health",
    summary="Service health and feed status",
)
async def get_health(db: Session = Depends(get_db)):
    """
    Returns service health, database status, last event time per store,
    and STALE_FEED warning if any store feed is > 10 minutes old.
    """
    result = compute_health(db)
    status_code = 200 if result["status"] == "OK" else 503
    return JSONResponse(content=result, status_code=status_code)


# ── POS loader endpoint (admin) ───────────────────────────────────────────────

@app.post(
    "/admin/load-pos",
    summary="Load POS transactions from a CSV file path (admin only)",
    include_in_schema=False,
)
async def admin_load_pos(
    request: Request,
    db: Session = Depends(get_db),
):
    body = await request.json()
    csv_path = body.get("csv_path")
    store_id = body.get("store_id", "STORE_BLR_002")
    if not csv_path:
        raise HTTPException(status_code=400, detail="csv_path is required.")
    n = load_pos_csv(csv_path, db, store_id)
    return {"loaded": n, "store_id": store_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
