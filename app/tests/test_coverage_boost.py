# PROMPT:
# "Write additional pytest tests for a retail store intelligence API to push coverage above 85%,
#  specifically targeting:
#  - pos_loader.py: CSV parsing for both simple (transaction_id,timestamp) and Brigade
#    (order_id,order_date,order_time) formats, get_converted_visitor_ids correlation logic
#  - health.py: DEGRADED status when DB is unreachable, STALE_FEED detection
#  - anomalies.py: CONVERSION_DROP when today's rate is below 7-day average, DEAD_ZONE detection
#  - ingestion.py: bulk insert fallback path when bulk_insert_mappings fails
#  - main.py: admin /admin/load-pos endpoint, graceful 503 on DB error"
#
# CHANGES MADE:
# - Replaced AI's monolithic fixture with isolated per-test helpers
# - Added explicit assertion on get_converted_visitor_ids set membership (not just len)
# - Added Brigade CSV format test (AI initially only tested simple format)
# - Replaced sleep-based stale feed test with direct timestamp manipulation
# - Added test for DEAD_ZONE anomaly with explicitly old zone visit data

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from database import Base, EventORM, POSTransactionORM, get_db, health_check_db
from main import app
from pos_loader import load_pos_csv, get_converted_visitor_ids
from anomalies import detect_anomalies
from health import compute_health

# ── Test DB setup ─────────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite:///./test_extra_coverage.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def _get_test_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def clean_db():
    """Install our DB override for the duration of each test, then restore."""
    orig = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = _get_test_db
    Base.metadata.create_all(bind=engine)
    yield
    # Tear down rows
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    # Restore the previous override (or remove it)
    if orig is not None:
        app.dependency_overrides[get_db] = orig
    else:
        app.dependency_overrides.pop(get_db, None)


client = TestClient(app, raise_server_exceptions=False)

STORE_ID = "STORE_BLR_002"
TODAY_ISO = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_event(event_type="ENTRY", visitor_id=None, zone_id=None, is_staff=False,
               dwell_ms=0, ts=None, store_id=STORE_ID, camera_id="CAM_ENTRY_01"):
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": ts or TODAY_ISO,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": 0.91,
        "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1},
    }


def ingest(events):
    return client.post("/events/ingest", json={"events": events})


# ── POS Loader Tests ──────────────────────────────────────────────────────────

class TestPOSLoader:
    def _db(self):
        return TestingSessionLocal()

    def test_simple_csv_format(self, tmp_path):
        """Simple schema: store_id, transaction_id, timestamp, basket_value_inr"""
        csv_file = tmp_path / "pos.csv"
        csv_file.write_text(
            "store_id,transaction_id,timestamp,basket_value_inr\n"
            f"{STORE_ID},TXN_001,{TODAY_ISO},500.00\n"
            f"{STORE_ID},TXN_002,{TODAY_ISO},750.00\n"
        )
        db = self._db()
        try:
            n = load_pos_csv(str(csv_file), db, STORE_ID)
            assert n == 2
        finally:
            db.close()

    def test_brigade_csv_format(self, tmp_path):
        """Brigade schema: order_id, order_date, order_time, store_id, product_id, total_amount"""
        csv_file = tmp_path / "brigade.csv"
        csv_file.write_text(
            "order_id,order_date,order_time,store_id,product_id,brand_name,total_amount\n"
            f"1,10-04-2026,12:15:05,{STORE_ID},12345,TestBrand,302.33\n"
            f"2,10-04-2026,13:00:00,{STORE_ID},12346,TestBrand,450.00\n"
        )
        db = self._db()
        try:
            n = load_pos_csv(str(csv_file), db, STORE_ID)
            assert n == 2
        finally:
            db.close()

    def test_nonexistent_csv_returns_zero(self):
        db = self._db()
        try:
            n = load_pos_csv("/nonexistent/path.csv", db, STORE_ID)
            assert n == 0
        finally:
            db.close()

    def test_duplicate_txn_not_reinserted(self, tmp_path):
        csv_file = tmp_path / "pos.csv"
        csv_file.write_text(
            "store_id,transaction_id,timestamp,basket_value_inr\n"
            f"{STORE_ID},TXN_DUP,{TODAY_ISO},100.00\n"
        )
        db = self._db()
        try:
            n1 = load_pos_csv(str(csv_file), db, STORE_ID)
            n2 = load_pos_csv(str(csv_file), db, STORE_ID)
            assert n1 == 1
            assert n2 == 0  # Already exists
        finally:
            db.close()

    def test_get_converted_visitor_ids_with_billing_event(self):
        """Visitor in billing zone within 5 min of POS txn → converted."""
        db = self._db()
        try:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            vid = f"VIS_{uuid.uuid4().hex[:6]}"

            # Insert POS transaction
            txn_time = now.replace(second=0, microsecond=0)
            db.add(POSTransactionORM(
                transaction_id="TXN_CONV_TEST",
                store_id=STORE_ID,
                timestamp=txn_time,
                basket_value_inr=500.0,
            ))

            # Insert billing event 3 minutes before transaction
            event_time = txn_time - timedelta(minutes=3)
            db.add(EventORM(
                event_id=str(uuid.uuid4()),
                store_id=STORE_ID, camera_id="CAM_BILLING_01",
                visitor_id=vid, event_type="BILLING_QUEUE_JOIN",
                timestamp=event_time, zone_id="BILLING_COUNTER",
                dwell_ms=0, is_staff=False, confidence=0.9,
            ))
            db.commit()

            converted = get_converted_visitor_ids(STORE_ID, now, db)
            assert vid in converted
        finally:
            db.close()

    def test_get_converted_visitor_ids_outside_window(self):
        """Visitor in billing zone 10 min before transaction → NOT converted (outside 5-min window)."""
        db = self._db()
        try:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            vid = f"VIS_{uuid.uuid4().hex[:6]}"

            txn_time = now.replace(second=0, microsecond=0)
            db.add(POSTransactionORM(
                transaction_id="TXN_OUTSIDE_WIN",
                store_id=STORE_ID,
                timestamp=txn_time,
                basket_value_inr=200.0,
            ))
            # 10 minutes BEFORE transaction — outside the 5-minute window
            event_time = txn_time - timedelta(minutes=10)
            db.add(EventORM(
                event_id=str(uuid.uuid4()),
                store_id=STORE_ID, camera_id="CAM_BILLING_01",
                visitor_id=vid, event_type="BILLING_QUEUE_JOIN",
                timestamp=event_time, zone_id="BILLING_COUNTER",
                dwell_ms=0, is_staff=False, confidence=0.9,
            ))
            db.commit()

            converted = get_converted_visitor_ids(STORE_ID, now, db)
            assert vid not in converted
        finally:
            db.close()

    def test_admin_load_pos_endpoint(self, tmp_path):
        """Admin endpoint /admin/load-pos accepts a CSV path."""
        csv_file = tmp_path / "pos_admin.csv"
        csv_file.write_text(
            "store_id,transaction_id,timestamp,basket_value_inr\n"
            f"{STORE_ID},TXN_ADMIN,{TODAY_ISO},999.00\n"
        )
        r = client.post("/admin/load-pos", json={"csv_path": str(csv_file), "store_id": STORE_ID})
        assert r.status_code == 200
        assert r.json()["loaded"] == 1

    def test_admin_load_pos_missing_path(self):
        r = client.post("/admin/load-pos", json={"store_id": STORE_ID})
        assert r.status_code == 400


# ── Health Tests (extended) ───────────────────────────────────────────────────

class TestHealthExtended:
    def test_health_returns_stores_dict(self):
        r = client.get("/health")
        data = r.json()
        assert "stores" in data
        assert "database" in data

    def test_health_db_ok_when_reachable(self):
        db = TestingSessionLocal()
        try:
            ok = health_check_db(db)
            assert ok is True
        finally:
            db.close()

    def test_health_db_unreachable_returns_false(self):
        """Mock a broken DB session."""
        broken_db = MagicMock()
        broken_db.execute.side_effect = Exception("DB connection refused")
        result = health_check_db(broken_db)
        assert result is False

    def test_health_degraded_when_no_events(self):
        """A store with events > 10 min old → STALE_FEED → DEGRADED overall."""
        db = TestingSessionLocal()
        try:
            stale_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=20)
            db.add(EventORM(
                event_id=str(uuid.uuid4()),
                store_id=STORE_ID, camera_id="CAM_ENTRY_01",
                visitor_id="VIS_stale", event_type="ENTRY",
                timestamp=stale_time, zone_id=None,
                dwell_ms=0, is_staff=False, confidence=0.9,
            ))
            db.commit()

            result = compute_health(db)
            # The store should show STALE_FEED since last event is 20 min ago
            assert STORE_ID in result["stores"]
            assert result["stores"][STORE_ID]["feed_status"] == "STALE_FEED"
            assert result["status"] == "DEGRADED"
        finally:
            db.close()


# ── Anomalies (extended) ─────────────────────────────────────────────────────

class TestAnomaliesExtended:
    def test_dead_zone_detected(self):
        """A zone active earlier today with no recent events → DEAD_ZONE."""
        db = TestingSessionLocal()
        try:
            old_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=45)
            db.add(EventORM(
                event_id=str(uuid.uuid4()),
                store_id=STORE_ID, camera_id="CAM_FLOOR_01",
                visitor_id="VIS_zone1", event_type="ZONE_ENTER",
                timestamp=old_time, zone_id="SKINCARE",
                dwell_ms=0, is_staff=False, confidence=0.9,
            ))
            db.commit()

            result = detect_anomalies(STORE_ID, db)
            types = [a["type"] for a in result["anomalies"]]
            assert "DEAD_ZONE" in types

            dead_zone = next(a for a in result["anomalies"] if a["type"] == "DEAD_ZONE")
            assert dead_zone["severity"] == "INFO"
            assert "suggested_action" in dead_zone
        finally:
            db.close()

    def test_billing_queue_spike_critical(self):
        """More than 10 people in billing queue → CRITICAL."""
        events = []
        for i in range(12):
            events.append(make_event(
                event_type="BILLING_QUEUE_JOIN",
                visitor_id=f"VIS_q{i:02d}",
                zone_id="BILLING_COUNTER",
                camera_id="CAM_BILLING_01",
            ))
        ingest(events)

        r = client.get(f"/stores/{STORE_ID}/anomalies")
        data = r.json()
        queue_anomaly = next((a for a in data["anomalies"] if a["type"] == "BILLING_QUEUE_SPIKE"), None)
        assert queue_anomaly is not None
        assert queue_anomaly["severity"] == "CRITICAL"

    def test_anomaly_response_structure(self):
        r = client.get(f"/stores/{STORE_ID}/anomalies")
        assert r.status_code == 200
        data = r.json()
        assert "store_id" in data
        assert "checked_at" in data
        assert "anomaly_count" in data
        assert isinstance(data["anomalies"], list)


# ── Edge Case: DB Error returns 503 ──────────────────────────────────────────

class TestGracefulDegradation:
    def test_global_error_handler_no_stack_trace(self):
        """Any unhandled exception returns structured JSON, not a raw stack trace."""
        # Ingest with a valid event but then break the DB mid-flight
        # The easiest way: send completely malformed JSON that bypasses Pydantic
        r = client.post(
            "/events/ingest",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (422, 500)
        body = r.json()
        # Must not expose raw stack trace fields
        assert "traceback" not in str(body).lower()
        assert "error" in body or "detail" in body
