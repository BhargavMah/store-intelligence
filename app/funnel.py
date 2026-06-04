"""
funnel.py — Conversion funnel computation.

GET /stores/{store_id}/funnel

Session is the unit of analysis (not raw events).
Re-entries do NOT create a new funnel entry for the same visitor_id on the same day.

Funnel stages:
  ENTRY → ZONE_VISIT → BILLING_QUEUE → PURCHASE
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import EventORM
from pos_loader import get_converted_visitor_ids


def compute_funnel(store_id: str, db: Session, date: datetime | None = None) -> dict:
    if date is None:
        date = datetime.now(timezone.utc).replace(tzinfo=None)

    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = date.replace(hour=23, minute=59, second=59)

    base = db.query(EventORM).filter(
        EventORM.store_id == store_id,
        EventORM.is_staff == False,
        EventORM.timestamp >= day_start,
        EventORM.timestamp <= day_end,
    )

    # ── Stage 1: Unique visitors who entered (deduplicated by visitor_id) ──
    entered_visitors = set(
        r.visitor_id
        for r in base.filter(
            EventORM.event_type.in_(["ENTRY", "REENTRY"])
        ).with_entities(EventORM.visitor_id).distinct().all()
    )
    entry_count = len(entered_visitors)

    # ── Stage 2: Visitors who visited at least one non-entry zone ─────────
    zone_visitors = set(
        r.visitor_id
        for r in base.filter(
            EventORM.event_type == "ZONE_ENTER",
            EventORM.zone_id.isnot(None),
            ~EventORM.zone_id.like("%ENTRY%"),
        ).with_entities(EventORM.visitor_id).distinct().all()
    )
    zone_visitors &= entered_visitors   # Only count people who actually entered
    zone_count = len(zone_visitors)

    # ── Stage 3: Visitors who joined billing queue ─────────────────────────
    billing_visitors = set(
        r.visitor_id
        for r in base.filter(
            EventORM.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
            EventORM.zone_id.like("%BILLING%"),
        ).with_entities(EventORM.visitor_id).distinct().all()
    )
    billing_visitors &= entered_visitors
    billing_count = len(billing_visitors)

    # ── Stage 4: Converted (POS correlation) ─────────────────────────────
    converted_ids = get_converted_visitor_ids(store_id, date, db)
    converted_ids &= entered_visitors
    purchase_count = len(converted_ids)

    # ── Drop-off percentages ──────────────────────────────────────────────
    def pct(n: int, total: int) -> float:
        return round(n / total * 100, 1) if total > 0 else 0.0

    def dropoff_pct(lost: int, from_count: int) -> float:
        return round(lost / from_count * 100, 1) if from_count > 0 else 0.0

    stages = [
        {"stage": "ENTRY",         "count": entry_count,   "pct": 100.0},
        {"stage": "ZONE_VISIT",    "count": zone_count,    "pct": pct(zone_count, entry_count)},
        {"stage": "BILLING_QUEUE", "count": billing_count, "pct": pct(billing_count, entry_count)},
        {"stage": "PURCHASE",      "count": purchase_count,"pct": pct(purchase_count, entry_count)},
    ]

    dropoffs = [
        {
            "from": "ENTRY",
            "to": "ZONE_VISIT",
            "lost": entry_count - zone_count,
            "lost_pct": dropoff_pct(entry_count - zone_count, entry_count),
        },
        {
            "from": "ZONE_VISIT",
            "to": "BILLING_QUEUE",
            "lost": zone_count - billing_count,
            "lost_pct": dropoff_pct(zone_count - billing_count, zone_count),
        },
        {
            "from": "BILLING_QUEUE",
            "to": "PURCHASE",
            "lost": billing_count - purchase_count,
            "lost_pct": dropoff_pct(billing_count - purchase_count, billing_count),
        },
    ]

    return {
        "store_id": store_id,
        "date": date.strftime("%Y-%m-%d"),
        "stages": stages,
        "dropoff": dropoffs,
        "note": "Session is the unit of analysis. Re-entries do not double-count a visitor.",
    }
