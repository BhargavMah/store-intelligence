"""
heatmap.py — Zone visit frequency and dwell heatmap.

GET /stores/{store_id}/heatmap

Returns zone visit frequency + avg dwell normalised 0–100.
Flags data_confidence="LOW" if fewer than 20 sessions in window.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import EventORM

LOW_CONFIDENCE_THRESHOLD = 20


def compute_heatmap(store_id: str, db: Session, date: datetime | None = None) -> dict:
    if date is None:
        date = datetime.now(timezone.utc).replace(tzinfo=None)

    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = date.replace(hour=23, minute=59, second=59)

    # ── Zone visit frequency ──────────────────────────────────────────────
    visit_rows = (
        db.query(
            EventORM.zone_id,
            func.count(EventORM.event_id).label("visit_count"),
        )
        .filter(
            EventORM.store_id == store_id,
            EventORM.is_staff == False,
            EventORM.event_type == "ZONE_ENTER",
            EventORM.zone_id.isnot(None),
            ~EventORM.zone_id.like("%ENTRY%"),
            EventORM.timestamp >= day_start,
            EventORM.timestamp <= day_end,
        )
        .group_by(EventORM.zone_id)
        .all()
    )

    # ── Avg dwell per zone ────────────────────────────────────────────────
    dwell_rows = (
        db.query(
            EventORM.zone_id,
            func.avg(EventORM.dwell_ms).label("avg_dwell_ms"),
        )
        .filter(
            EventORM.store_id == store_id,
            EventORM.is_staff == False,
            EventORM.event_type.in_(["ZONE_EXIT", "ZONE_DWELL"]),
            EventORM.zone_id.isnot(None),
            EventORM.dwell_ms > 0,
            EventORM.timestamp >= day_start,
            EventORM.timestamp <= day_end,
        )
        .group_by(EventORM.zone_id)
        .all()
    )

    dwell_by_zone = {r.zone_id: r.avg_dwell_ms for r in dwell_rows}

    # ── Total session count ───────────────────────────────────────────────
    total_sessions = (
        db.query(EventORM.visitor_id)
        .filter(
            EventORM.store_id == store_id,
            EventORM.is_staff == False,
            EventORM.event_type.in_(["ENTRY", "REENTRY"]),
            EventORM.timestamp >= day_start,
            EventORM.timestamp <= day_end,
        )
        .distinct()
        .count()
    )

    overall_confidence = "LOW" if total_sessions < LOW_CONFIDENCE_THRESHOLD else "HIGH"

    # ── Normalise visit frequency 0–100 ──────────────────────────────────
    if not visit_rows:
        return {
            "store_id": store_id,
            "date": date.strftime("%Y-%m-%d"),
            "zones": [],
            "total_sessions": total_sessions,
            "data_confidence": overall_confidence,
        }

    counts = [r.visit_count for r in visit_rows]
    min_c, max_c = min(counts), max(counts)

    def normalise(val: int) -> int:
        if max_c == min_c:
            return 50
        return round((val - min_c) / (max_c - min_c) * 100)

    zones_out = []
    for row in sorted(visit_rows, key=lambda r: r.visit_count, reverse=True):
        avg_dwell_s = dwell_by_zone.get(row.zone_id)
        avg_dwell_s = round(avg_dwell_s / 1000, 2) if avg_dwell_s else None

        zone_confidence = "LOW" if row.visit_count < LOW_CONFIDENCE_THRESHOLD else "HIGH"

        zones_out.append({
            "zone_id": row.zone_id,
            "visit_frequency": row.visit_count,
            "avg_dwell_seconds": avg_dwell_s,
            "normalised_score": normalise(row.visit_count),
            "data_confidence": zone_confidence,
        })

    return {
        "store_id": store_id,
        "date": date.strftime("%Y-%m-%d"),
        "zones": zones_out,
        "total_sessions": total_sessions,
        "data_confidence": overall_confidence,
    }
