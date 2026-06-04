import time
import uuid
import random
import requests
from datetime import datetime, timezone

API_URL = "http://localhost:8000/events/ingest"
STORE_ID = "STORE_BLR_002"

EVENT_TYPES = ["ENTRY", "ZONE_ENTER", "ZONE_EXIT", "BILLING_QUEUE_JOIN", "EXIT"]
ZONES = ["SKINCARE", "MAKEUP", "FRAGRANCE", "BILLING"]

print("Starting live feed simulation for Dashboard...")
while True:
    try:
        event = {
            "event_id": str(uuid.uuid4()),
            "store_id": STORE_ID,
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": f"VIS_{random.randint(1000, 9999)}",
            "event_type": random.choice(EVENT_TYPES),
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "zone_id": random.choice(ZONES),
            "dwell_ms": random.randint(1000, 50000),
            "is_staff": False,
            "confidence": round(random.uniform(0.7, 0.99), 2),
            "metadata": {
                "queue_depth": random.randint(0, 15) if random.random() > 0.5 else 0,
                "session_seq": 1
            }
        }
        
        # If queue_depth > 10, it'll trigger a CRITICAL anomaly!
        
        r = requests.post(API_URL, json={"events": [event]})
        print(f"Sent {event['event_type']} -> HTTP {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")
        
    time.sleep(1.5)
