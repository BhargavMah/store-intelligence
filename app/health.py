"""
health.py — Service health endpoint.

GET /health

Returns:
  - service status
  - last event timestamp per store
  - STALE_FEED warning if > 10 min lag
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import EventORM, health_check_db

STALE_FEED_MINUTES = 10


def compute_health(db: Session) -> dict:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    db_ok = health_check_db(db)

    if not db_ok:
        return {
            "status": "DEGRADED",
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "database": "UNREACHABLE",
            "stores": {},
        }

    # ── Per-store stats ───────────────────────────────────────────────────
    store_stats_rows = (
        db.query(
            EventORM.store_id,
            func.max(EventORM.timestamp).label("last_event_at"),
            func.count(EventORM.event_id).label("total_events"),
        )
        .filter(EventORM.timestamp >= now.replace(hour=0, minute=0, second=0))
        .group_by(EventORM.store_id)
        .all()
    )

    stores: dict[str, dict] = {}
    overall_status = "OK"

    for row in store_stats_rows:
        lag_s = (now - row.last_event_at).total_seconds()
        if lag_s > STALE_FEED_MINUTES * 60:
            feed_status = "STALE_FEED"
            overall_status = "DEGRADED"
        else:
            feed_status = "LIVE"

        stores[row.store_id] = {
            "last_event_at": row.last_event_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feed_status": feed_status,
            "event_count_today": row.total_events,
            "lag_seconds": round(lag_s, 1),
        }

    return {
        "status": overall_status,
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "database": "OK",
        "stores": stores,
    }
