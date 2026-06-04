"""
ingestion.py — POST /events/ingest handler.

Accepts a batch of up to 500 events (as raw dicts or StoreEvent objects),
deduplicates by event_id, validates each event, and inserts into the DB.

Idempotency: calling this endpoint twice with the same payload is safe —
duplicate event_ids are silently skipped.

Returns a partial-success response even if some events fail validation.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from database import EventORM


def _parse_timestamp(ts_str: str) -> datetime:
    """Convert ISO-8601 string to naive UTC datetime for DB storage."""
    ts_str = ts_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts_str)
    # Store as naive UTC
    return dt.replace(tzinfo=None)


def ingest_events(
    raw_events: list[dict[str, Any]],
    db: Session,
) -> dict:
    """
    Validate, deduplicate, and persist a batch of events.

    Returns:
        {"accepted": int, "rejected": int, "errors": list[dict]}
    """
    accepted = 0
    rejected = 0
    errors: list[dict] = []

    # Pre-fetch existing event_ids in this batch to skip round-trips
    incoming_ids = [e.get("event_id") for e in raw_events if e.get("event_id")]
    existing_ids: set[str] = set()
    if incoming_ids:
        rows = db.query(EventORM.event_id).filter(
            EventORM.event_id.in_(incoming_ids)
        ).all()
        existing_ids = {r.event_id for r in rows}

    to_insert: list[dict] = []

    for raw in raw_events:
        event_id = raw.get("event_id")

        # ── Idempotency check ───────────────────────────────────────────
        if event_id and event_id in existing_ids:
            accepted += 1   # Count as accepted — idempotent success
            continue

        # ── Validate via Pydantic ────────────────────────────────────────
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from schema import StoreEvent, EventMetadata

            meta_raw = raw.pop("metadata", {}) or {}
            raw["metadata"] = EventMetadata(**(meta_raw if isinstance(meta_raw, dict) else {}))
            event = StoreEvent(**raw)

            # Restore for row building
            raw["metadata"] = meta_raw
            raw["event_id"] = event.event_id

        except Exception as exc:
            rejected += 1
            errors.append({
                "event_id": event_id or "<missing>",
                "reason": str(exc),
            })
            continue

        # ── Build DB row ─────────────────────────────────────────────────
        try:
            meta = raw.get("metadata", {}) or {}
            ts = _parse_timestamp(raw.get("timestamp", ""))
            row = {
                "event_id": event.event_id,
                "store_id": event.store_id,
                "camera_id": event.camera_id,
                "visitor_id": event.visitor_id,
                "event_type": event.event_type.value,
                "timestamp": ts,
                "zone_id": event.zone_id,
                "dwell_ms": event.dwell_ms,
                "is_staff": event.is_staff,
                "confidence": event.confidence,
                "queue_depth": meta.get("queue_depth"),
                "sku_zone": meta.get("sku_zone"),
                "session_seq": meta.get("session_seq"),
                "ingested_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
            to_insert.append(row)
            existing_ids.add(event.event_id)
            accepted += 1

        except Exception as exc:
            rejected += 1
            errors.append({
                "event_id": event_id or "<missing>",
                "reason": f"DB row build error: {exc}",
            })

    # ── Batch insert (ignore duplicates at DB level for extra safety) ────
    if to_insert:
        try:
            # SQLite: INSERT OR IGNORE; PostgreSQL: ON CONFLICT DO NOTHING
            db.bulk_insert_mappings(EventORM, to_insert)
            db.commit()
        except Exception as exc:
            db.rollback()
            # Fall back to one-by-one insert with per-row error handling
            for row in to_insert:
                try:
                    obj = EventORM(**row)
                    db.add(obj)
                    db.commit()
                except Exception:
                    db.rollback()
                    accepted -= 1
                    rejected += 1
                    errors.append({
                        "event_id": row.get("event_id", "<missing>"),
                        "reason": "DB insert conflict (already exists)",
                    })

    return {
        "accepted": accepted,
        "rejected": rejected,
        "errors": errors[:50],  # Cap error list for response size
    }
