"""
pos_loader.py — Load POS transactions from CSV and correlate with visitor sessions.

Correlation rule:
  A visitor is "converted" if they have a ZONE_ENTER / BILLING_QUEUE_JOIN event
  in the billing zone within 5 minutes BEFORE a POS transaction timestamp for
  the same store_id on the same day.

Since POS data has no customer_id, this is probabilistic: we assign conversion
credit to visitors who were in the billing zone in the look-back window.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from database import POSTransactionORM

CONVERSION_WINDOW_MINUTES = 5
BILLING_ZONE_TYPES = {"BILLING"}


def load_pos_csv(csv_path: str, db: Session, store_id: str) -> int:
    """
    Load POS transactions from a CSV file into the database.
    
    Expected CSV columns (flexible — detects format automatically):
    - Simple: store_id, transaction_id, timestamp, basket_value_inr
    - Full (Brigade data): order_id, order_date, order_time, store_id, product_id, ...
    
    Returns number of rows inserted.
    """
    inserted = 0
    path = Path(csv_path)
    if not path.exists():
        return 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows_to_insert = []

        for row in reader:
            # ── Detect and normalise schema ──────────────────────────────
            try:
                sid = row.get("store_id", store_id)

                # Simple schema: transaction_id, timestamp
                if "transaction_id" in row and "timestamp" in row:
                    txn_id = row["transaction_id"]
                    ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
                    amount = float(row.get("basket_value_inr", 0) or 0)
                    brand = None
                    product_id = None
                    order_date = None
                    order_time = None

                # Full Brigade schema: order_id, order_date, order_time
                elif "order_id" in row and "order_date" in row:
                    txn_id = f"TXN_{row['order_id']}_{row.get('product_id', '')}"
                    order_date = row.get("order_date", "")
                    order_time = row.get("order_time", "")
                    amount = float(row.get("total_amount", 0) or 0)
                    brand = row.get("brand_name")
                    product_id = row.get("product_id")
                    # Parse DD-MM-YYYY HH:MM:SS
                    try:
                        ts = datetime.strptime(f"{order_date} {order_time}", "%d-%m-%Y %H:%M:%S")
                    except ValueError:
                        ts = datetime.strptime(f"{order_date} {order_time}", "%Y-%m-%d %H:%M:%S")

                else:
                    continue

                rows_to_insert.append({
                    "transaction_id": txn_id,
                    "store_id": sid,
                    "timestamp": ts,
                    "basket_value_inr": amount,
                    "brand_name": brand,
                    "product_id": str(product_id) if product_id else None,
                    "order_date": order_date,
                    "order_time": order_time,
                })

            except (ValueError, KeyError):
                continue

    # Bulk insert, ignore duplicates
    for row_data in rows_to_insert:
        existing = db.get(POSTransactionORM, row_data["transaction_id"])
        if not existing:
            db.add(POSTransactionORM(**row_data))
            inserted += 1

    db.commit()
    return inserted


def get_converted_visitor_ids(
    store_id: str,
    date: datetime,
    db: Session,
) -> set[str]:
    """
    Return a set of visitor_ids who can be classified as "converted" on the given date.
    
    Logic: for each POS transaction in the store on this date, find visitor_ids
    who had a billing-zone event within CONVERSION_WINDOW_MINUTES before the transaction.
    """
    from database import EventORM

    date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    date_end = date.replace(hour=23, minute=59, second=59)

    # Get all POS transactions for this store+date
    txns = db.query(POSTransactionORM).filter(
        POSTransactionORM.store_id == store_id,
        POSTransactionORM.timestamp >= date_start,
        POSTransactionORM.timestamp <= date_end,
    ).all()

    converted: set[str] = set()

    # For each transaction, find visitors in billing zone within look-back window
    for txn in txns:
        window_start = txn.timestamp - timedelta(minutes=CONVERSION_WINDOW_MINUTES)
        window_end = txn.timestamp

        billing_visitors = db.query(EventORM.visitor_id).filter(
            EventORM.store_id == store_id,
            EventORM.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
            EventORM.zone_id.like("%BILLING%"),
            EventORM.is_staff == False,
            EventORM.timestamp >= window_start,
            EventORM.timestamp <= window_end,
        ).distinct().all()

        for (vid,) in billing_visitors:
            converted.add(vid)

    return converted
