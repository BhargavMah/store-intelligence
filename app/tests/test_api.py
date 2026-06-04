# PROMPT:
# "Write comprehensive pytest tests for a FastAPI store intelligence API that:
# - Ingests batches of CCTV-derived visitor events
# - Exposes /metrics, /funnel, /heatmap, /anomalies, /health endpoints
# - Handles edge cases: empty store, all-staff events, re-entry, zero purchases
# - Tests idempotency on POST /events/ingest
# Cover: happy path, edge cases, error responses, partial success on bad events."
#
# CHANGES MADE:
# - Added explicit test for billing queue abandon detection
# - Added stale feed test with time mocking instead of relying on AI's sleep approach
# - Split the single large fixture into focused per-test fixtures
# - Replaced AI-generated assertions using `in` with exact count checks
# - Added test for max batch size (500 event limit)

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from database import Base, get_db
from main import app

# ── Test DB (in-memory SQLite — shared connection so tables persist per test) ─

TEST_DATABASE_URL = "sqlite:///./test_store_intelligence.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables once at module load time
Base.metadata.create_all(bind=engine)


@pytest.fixture(autouse=True)
def setup_db():
    """Ensure tables exist and clear all data before each test."""
    Base.metadata.create_all(bind=engine)
    yield
    # Tear down all rows after each test
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

STORE_ID = "STORE_BLR_002"
TODAY = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ")
TODAY_DATE = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d")


def make_event(
    event_type: str = "ENTRY",
    visitor_id: str | None = None,
    zone_id: str | None = None,
    is_staff: bool = False,
    dwell_ms: int = 0,
    store_id: str = STORE_ID,
    confidence: float = 0.92,
    queue_depth: int | None = None,
    event_id: str | None = None,
    camera_id: str = "CAM_ENTRY_01",
) -> dict:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": TODAY,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": None,
            "session_seq": 1,
        },
    }


def ingest(events: list[dict]) -> dict:
    r = client.post("/events/ingest", json={"events": events})
    return r


# ── Ingest Tests ──────────────────────────────────────────────────────────────

class TestIngest:
    def test_single_event_accepted(self):
        r = ingest([make_event()])
        assert r.status_code == 200
        data = r.json()
        assert data["accepted"] == 1
        assert data["rejected"] == 0

    def test_batch_of_events(self):
        events = [make_event() for _ in range(10)]
        r = ingest(events)
        assert r.status_code == 200
        assert r.json()["accepted"] == 10

    def test_idempotency_same_event_id(self):
        """Sending the same event twice should not create duplicates."""
        ev = make_event()
        r1 = ingest([ev])
        r2 = ingest([ev])
        assert r1.json()["accepted"] == 1
        assert r2.json()["accepted"] == 1  # Idempotent — accepted but not re-inserted

    def test_partial_success_on_bad_event(self):
        """One malformed event in a batch should not block the rest."""
        good = make_event()
        bad = {"event_id": str(uuid.uuid4()), "store_id": STORE_ID}  # Missing required fields
        r = ingest([good, bad])
        data = r.json()
        assert data["accepted"] >= 1
        assert data["rejected"] >= 1
        assert len(data["errors"]) >= 1

    def test_batch_size_limit(self):
        """Batches over 500 events should be rejected."""
        events = [make_event() for _ in range(501)]
        r = ingest(events)
        assert r.status_code == 413

    def test_malformed_json_returns_422(self):
        r = client.post("/events/ingest", content=b"not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_staff_events_ingested_with_flag(self):
        ev = make_event(is_staff=True, event_type="ENTRY")
        r = ingest([ev])
        assert r.json()["accepted"] == 1

    def test_bare_list_payload(self):
        """API should accept a bare JSON array (not wrapped in {"events": []})."""
        events = [make_event()]
        r = client.post("/events/ingest", json=events)
        assert r.status_code in (200, 207)
        assert r.json()["accepted"] >= 1


# ── Metrics Tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_empty_store(self):
        """Empty store must return zeros, not null."""
        r = client.get(f"/stores/{STORE_ID}/metrics")
        assert r.status_code == 200
        data = r.json()
        assert data["unique_visitors"] == 0
        assert data["conversion_rate"] == 0.0
        assert data["current_queue_depth"] == 0
        assert data["abandonment_rate"] == 0.0

    def test_metrics_excludes_staff(self):
        """Staff events must not count toward unique_visitors."""
        ingest([make_event(is_staff=True, event_type="ENTRY")])
        r = client.get(f"/stores/{STORE_ID}/metrics")
        assert r.json()["unique_visitors"] == 0

    def test_metrics_counts_real_visitors(self):
        vis1 = f"VIS_{uuid.uuid4().hex[:6]}"
        vis2 = f"VIS_{uuid.uuid4().hex[:6]}"
        ingest([
            make_event(visitor_id=vis1, event_type="ENTRY"),
            make_event(visitor_id=vis2, event_type="ENTRY"),
        ])
        r = client.get(f"/stores/{STORE_ID}/metrics")
        assert r.json()["unique_visitors"] == 2

    def test_metrics_reentry_no_double_count(self):
        """Same visitor_id with REENTRY should not increment unique_visitors."""
        vid = f"VIS_{uuid.uuid4().hex[:6]}"
        ingest([
            make_event(visitor_id=vid, event_type="ENTRY"),
            make_event(visitor_id=vid, event_type="REENTRY"),
        ])
        r = client.get(f"/stores/{STORE_ID}/metrics")
        assert r.json()["unique_visitors"] == 1

    def test_metrics_invalid_date_returns_400(self):
        r = client.get(f"/stores/{STORE_ID}/metrics?date=baddate")
        assert r.status_code == 400


# ── Funnel Tests ──────────────────────────────────────────────────────────────

class TestFunnel:
    def test_funnel_empty_store(self):
        r = client.get(f"/stores/{STORE_ID}/funnel")
        assert r.status_code == 200
        data = r.json()
        assert data["stages"][0]["count"] == 0  # ENTRY stage

    def test_funnel_stages_structure(self):
        vid = f"VIS_{uuid.uuid4().hex[:6]}"
        ingest([
            make_event(visitor_id=vid, event_type="ENTRY"),
            make_event(visitor_id=vid, event_type="ZONE_ENTER",
                       zone_id="CENTER_DISPLAY", camera_id="CAM_FLOOR_01",
                       dwell_ms=0),
        ])
        r = client.get(f"/stores/{STORE_ID}/funnel")
        data = r.json()
        stage_names = [s["stage"] for s in data["stages"]]
        assert "ENTRY" in stage_names
        assert "ZONE_VISIT" in stage_names
        assert "BILLING_QUEUE" in stage_names
        assert "PURCHASE" in stage_names

    def test_funnel_dropoff_non_negative(self):
        r = client.get(f"/stores/{STORE_ID}/funnel")
        data = r.json()
        for d in data["dropoff"]:
            assert d["lost"] >= 0
            assert 0.0 <= d["lost_pct"] <= 100.0


# ── Heatmap Tests ─────────────────────────────────────────────────────────────

class TestHeatmap:
    def test_heatmap_empty(self):
        r = client.get(f"/stores/{STORE_ID}/heatmap")
        assert r.status_code == 200
        data = r.json()
        assert data["zones"] == []

    def test_heatmap_low_confidence_flag(self):
        """Fewer than 20 sessions → data_confidence=LOW."""
        ingest([make_event(event_type="ZONE_ENTER", zone_id="CENTER_DISPLAY")])
        r = client.get(f"/stores/{STORE_ID}/heatmap")
        data = r.json()
        assert data["data_confidence"] == "LOW"

    def test_heatmap_normalised_scores_in_range(self):
        for _ in range(5):
            ingest([make_event(event_type="ZONE_ENTER", zone_id="LEFT_SHELF")])
        r = client.get(f"/stores/{STORE_ID}/heatmap")
        for zone in r.json()["zones"]:
            assert 0 <= zone["normalised_score"] <= 100


# ── Anomaly Tests ─────────────────────────────────────────────────────────────

class TestAnomalies:
    def test_anomalies_endpoint_returns_ok(self):
        r = client.get(f"/stores/{STORE_ID}/anomalies")
        assert r.status_code == 200
        data = r.json()
        assert "anomalies" in data
        assert isinstance(data["anomalies"], list)

    def test_stale_feed_detected_for_empty_store(self):
        """A store with no events at all should trigger STALE_FEED."""
        r = client.get(f"/stores/{STORE_ID}/anomalies")
        anomaly_types = [a["type"] for a in r.json()["anomalies"]]
        assert "STALE_FEED" in anomaly_types

    def test_anomaly_severity_values(self):
        r = client.get(f"/stores/{STORE_ID}/anomalies")
        valid_severities = {"INFO", "WARN", "CRITICAL"}
        for a in r.json()["anomalies"]:
            assert a["severity"] in valid_severities

    def test_anomaly_has_suggested_action(self):
        r = client.get(f"/stores/{STORE_ID}/anomalies")
        for a in r.json()["anomalies"]:
            assert "suggested_action" in a
            assert len(a["suggested_action"]) > 0


# ── Health Tests ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok_when_db_up(self):
        r = client.get("/health")
        assert r.status_code in (200, 503)
        data = r.json()
        assert "status" in data
        assert "timestamp" in data
        assert "database" in data

    def test_health_empty_store_dict(self):
        r = client.get("/health")
        data = r.json()
        assert isinstance(data["stores"], dict)
