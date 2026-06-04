"""
metrics.py — Real-time store metrics computation.

GET /stores/{store_id}/metrics

Returns:
  - unique_visitors (today, non-staff)
  - conversion_rate  (POS correlation)
  - avg_dwell_per_zone (seconds)
  - current_queue_depth
  - abandonment_rate
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import EventORM
from pos_loader import get_converted_visitor_ids


def compute_metrics(store_id: str, db: Session, date: datetime | None = None) -> dict:
    """Compute real-time store metrics for a given store and date (default: today)."""

    if date is None:
        date = datetime.now(timezone.utc).replace(tzinfo=None)

    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = date.replace(hour=23, minute=59, second=59)

    # ── Base query filter ─────────────────────────────────────────────────
    base_q = db.query(EventORM).filter(
        EventORM.store_id == store_id,
        EventORM.is_staff == False,
        EventORM.timestamp >= day_start,
        EventORM.timestamp <= day_end,
    )

    # ── Unique visitors (by visitor_id from ENTRY events) ─────────────────
    unique_visitors = (
        base_q.filter(EventORM.event_type == "ENTRY")
        .with_entities(EventORM.visitor_id)
        .distinct()
        .count()
    )

    # ── Conversion rate ───────────────────────────────────────────────────
    converted_ids = get_converted_visitor_ids(store_id, date, db)
    conversion_rate = (
        round(len(converted_ids) / unique_visitors, 4)
        if unique_visitors > 0
        else 0.0
    )

    # ── Avg dwell per zone ────────────────────────────────────────────────
    dwell_rows = (
        base_q.filter(
            EventORM.event_type.in_(["ZONE_EXIT", "ZONE_DWELL"]),
            EventORM.zone_id.isnot(None),
            EventORM.dwell_ms > 0,
        )
        .with_entities(EventORM.zone_id, func.avg(EventORM.dwell_ms).label("avg_ms"))
        .group_by(EventORM.zone_id)
        .all()
    )
    avg_dwell_per_zone = {
        row.zone_id: round(row.avg_ms / 1000, 2) for row in dwell_rows
    }

    # ── Current queue depth ───────────────────────────────────────────────
    # Look at BILLING_QUEUE_JOIN events in the last 15 minutes with no matching EXIT
    recent_cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=15)
    queue_joins = (
        db.query(EventORM.visitor_id)
        .filter(
            EventORM.store_id == store_id,
            EventORM.event_type == "BILLING_QUEUE_JOIN",
            EventORM.timestamp >= recent_cutoff,
            EventORM.is_staff == False,
        )
        .distinct()
        .all()
    )
    queue_visitor_ids = {r.visitor_id for r in queue_joins}

    # Remove those who have already had an EXIT from billing or from store
    exited = (
        db.query(EventORM.visitor_id)
        .filter(
            EventORM.store_id == store_id,
            EventORM.event_type.in_(["ZONE_EXIT", "EXIT", "BILLING_QUEUE_ABANDON"]),
            EventORM.timestamp >= recent_cutoff,
            EventORM.visitor_id.in_(queue_visitor_ids),
        )
        .distinct()
        .all()
    )
    exited_ids = {r.visitor_id for r in exited}
    current_queue_depth = max(0, len(queue_visitor_ids - exited_ids))

    # ── Abandonment rate ──────────────────────────────────────────────────
    total_queue_joins = (
        base_q.filter(EventORM.event_type == "BILLING_QUEUE_JOIN").count()
    )
    total_abandons = (
        base_q.filter(EventORM.event_type == "BILLING_QUEUE_ABANDON").count()
    )
    abandonment_rate = (
        round(total_abandons / total_queue_joins, 4)
        if total_queue_joins > 0
        else 0.0
    )

    return {
        "store_id": store_id,
        "date": date.strftime("%Y-%m-%d"),
        "unique_visitors": unique_visitors,
        "conversion_rate": conversion_rate,
        "avg_dwell_per_zone": avg_dwell_per_zone,
        "current_queue_depth": current_queue_depth,
        "abandonment_rate": abandonment_rate,
        "converted_visitors": len(converted_ids),
        "data_freshness": "REAL_TIME",
    }
