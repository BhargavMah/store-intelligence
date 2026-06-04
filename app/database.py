"""
database.py — SQLAlchemy engine, session factory, and table definitions.
Supports both SQLite (dev) and PostgreSQL (prod) via DATABASE_URL env var.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Connection ───────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite:///./store_intelligence.db"
)

# SQLite needs check_same_thread=False for FastAPI's async workers
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,   # detect stale connections gracefully
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── ORM Base + Models ────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class EventORM(Base):
    __tablename__ = "events"

    event_id = Column(String, primary_key=True, index=True)
    store_id = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False)
    visitor_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    zone_id = Column(String, nullable=True, index=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)
    queue_depth = Column(Integer, nullable=True)
    sku_zone = Column(String, nullable=True)
    session_seq = Column(Integer, nullable=True)
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class POSTransactionORM(Base):
    __tablename__ = "pos_transactions"

    transaction_id = Column(String, primary_key=True, index=True)
    store_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    basket_value_inr = Column(Float, default=0.0)
    order_date = Column(String, nullable=True)
    order_time = Column(String, nullable=True)
    product_id = Column(String, nullable=True)
    brand_name = Column(String, nullable=True)


# ── Init + dependency ────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def health_check_db(db: Session) -> bool:
    """Return True if the database is reachable."""
    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
