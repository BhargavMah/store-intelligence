"""
anomalies.py — Real-time operational anomaly detection.

GET /stores/{store_id}/anomalies

Detects:
  - BILLING_QUEUE_SPIKE: current queue depth exceeds threshold
  - CONVERSION_DROP: today's rate is significantly below 7-day average
  - DEAD_ZONE: no zone visits in the last 30 minutes
  - STALE_FEED: no events from store in last 10 minutes
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import EventORM
from pos_loader import get_converted_visitor_ids

# ── Thresholds (configurable via environment) ─────────────────────────────

QUEUE_SPIKE_THRESHOLD = int(5)        # Queue depth > 5 → WARN
QUEUE_CRITICAL_THRESHOLD = int(10)    # Queue depth > 10 → CRITICAL
CONVERSION_DROP_RATIO = 0.70          # Today < 70% of 7-day avg → anomaly
DEAD_ZONE_MINUTES = 30                # No zone visits in 30 min → anomaly
STALE_FEED_MINUTES = 10              # No events in 10 min → STALE_FEED


def detect_anomalies(store_id: str, db: Session) -> dict:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    anomalies: list[dict] = []

    # ── 1. Billing Queue Spike ─────────────────────────────────────────────
    recent_cutoff = now - timedelta(minutes=15)

    # Count visitors who joined billing queue recently and haven't exited
    queue_joins = set(
        r.visitor_id
        for r in db.query(EventORM.visitor_id)
        .filter(
            EventORM.store_id == store_id,
            EventORM.event_type == "BILLING_QUEUE_JOIN",
            EventORM.timestamp >= recent_cutoff,
            EventORM.is_staff == False,
        ).distinct().all()
    )
    queue_exits = set(
        r.visitor_id
        for r in db.query(EventORM.visitor_id)
        .filter(
            EventORM.store_id == store_id,
            EventORM.event_type.in_(["EXIT", "ZONE_EXIT", "BILLING_QUEUE_ABANDON"]),
            EventORM.zone_id.like("%BILLING%"),
            EventORM.timestamp >= recent_cutoff,
            EventORM.visitor_id.in_(queue_joins),
        ).distinct().all()
    )
    current_queue = max(0, len(queue_joins - queue_exits))

    if current_queue > QUEUE_CRITICAL_THRESHOLD:
        severity = "CRITICAL"
    elif current_queue > QUEUE_SPIKE_THRESHOLD:
        severity = "WARN"
    else:
        severity = None

    if severity:
        anomalies.append({
            "type": "BILLING_QUEUE_SPIKE",
            "severity": severity,
            "description": f"Queue depth {current_queue} (threshold: {QUEUE_SPIKE_THRESHOLD})",
            "suggested_action": "Open an additional billing counter immediately.",
            "detected_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "value": current_queue,
        })

    # ── 2. Conversion Drop vs 7-day average ───────────────────────────────
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _daily_rate(day_start: datetime) -> float:
        day_end = day_start.replace(hour=23, minute=59, second=59)
        visitors = (
            db.query(EventORM.visitor_id)
            .filter(
                EventORM.store_id == store_id,
                EventORM.event_type.in_(["ENTRY", "REENTRY"]),
                EventORM.is_staff == False,
                EventORM.timestamp >= day_start,
                EventORM.timestamp <= day_end,
            ).distinct().count()
        )
        if visitors == 0:
            return 0.0
        converted = get_converted_visitor_ids(store_id, day_start, db)
        return len(converted) / visitors

    today_rate = _daily_rate(today_start)

    # Compute 7-day rolling average (exclude today)
    historical_rates = []
    for days_ago in range(1, 8):
        past_day = today_start - timedelta(days=days_ago)
        r = _daily_rate(past_day)
        if r > 0:
            historical_rates.append(r)

    if historical_rates:
        avg_7d = sum(historical_rates) / len(historical_rates)
        if avg_7d > 0 and today_rate < avg_7d * CONVERSION_DROP_RATIO:
            drop_pct = round((1 - today_rate / avg_7d) * 100, 1)
            anomalies.append({
                "type": "CONVERSION_DROP",
                "severity": "CRITICAL" if drop_pct > 40 else "WARN",
                "description": (
                    f"Conversion rate {today_rate:.1%} is {drop_pct}% below "
                    f"7-day average {avg_7d:.1%}"
                ),
                "suggested_action": (
                    "Review billing staff levels, check active promotions, "
                    "inspect billing zone for issues."
                ),
                "detected_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "value": today_rate,
                "baseline_7d": avg_7d,
            })

    # ── 3. Dead Zone (no visits in last 30 minutes) ────────────────────────
    zone_cutoff = now - timedelta(minutes=DEAD_ZONE_MINUTES)

    # Get all zones that have EVER had events for this store
    all_zones = (
        db.query(EventORM.zone_id)
        .filter(
            EventORM.store_id == store_id,
            EventORM.event_type == "ZONE_ENTER",
            EventORM.zone_id.isnot(None),
            ~EventORM.zone_id.like("%ENTRY%"),
            EventORM.timestamp >= today_start,
        )
        .distinct()
        .all()
    )
    all_zone_ids = {r.zone_id for r in all_zones}

    # Zones with recent activity
    recent_zones = (
        db.query(EventORM.zone_id)
        .filter(
            EventORM.store_id == store_id,
            EventORM.event_type == "ZONE_ENTER",
            EventORM.timestamp >= zone_cutoff,
        )
        .distinct()
        .all()
    )
    recent_zone_ids = {r.zone_id for r in recent_zones}

    dead_zones = all_zone_ids - recent_zone_ids
    for zone_id in dead_zones:
        anomalies.append({
            "type": "DEAD_ZONE",
            "severity": "INFO",
            "description": (
                f"Zone '{zone_id}' has had no visitor activity "
                f"in the last {DEAD_ZONE_MINUTES} minutes."
            ),
            "suggested_action": (
                "Check zone signage, lighting, or product availability. "
                "Consider activating a promotion in this zone."
            ),
            "detected_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "zone_id": zone_id,
        })

    # ── 4. Stale Feed ─────────────────────────────────────────────────────
    stale_cutoff = now - timedelta(minutes=STALE_FEED_MINUTES)
    last_event = (
        db.query(func.max(EventORM.timestamp))
        .filter(EventORM.store_id == store_id)
        .scalar()
    )

    if last_event is None or last_event < stale_cutoff:
        lag_min = (
            round((now - last_event).total_seconds() / 60, 1)
            if last_event
            else None
        )
        anomalies.append({
            "type": "STALE_FEED",
            "severity": "CRITICAL",
            "description": (
                f"No events received from store {store_id} "
                f"in the last {lag_min or '?'} minutes."
            ),
            "suggested_action": (
                "Check camera connectivity and detection pipeline health. "
                "Restart the detection pipeline if needed."
            ),
            "detected_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_event_at": last_event.strftime("%Y-%m-%dT%H:%M:%SZ") if last_event else None,
        })

    return {
        "store_id": store_id,
        "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }
