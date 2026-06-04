import requests, json, uuid, sys, os

BASE = "http://localhost:8000"

# 1. Health check
r = requests.get(f"{BASE}/health")
print("HEALTH:", r.status_code)
print(json.dumps(r.json(), indent=2))

# 2. Load + normalise the provided sample events
events = []
sample_path = r"C:\Users\jatin\Downloads\Purplle_hack\sample_eventsbe42122.jsonl"
with open(sample_path) as f:
    for line in f:
        line = line.strip()
        if line:
            events.append(json.loads(line))

print(f"\nLoaded {len(events)} sample events")

type_map = {
    "entry": "ENTRY", "exit": "EXIT",
    "zone_entered": "ZONE_ENTER", "zone_exited": "ZONE_EXIT",
    "queue_completed": "BILLING_QUEUE_JOIN",
    "queue_abandoned": "BILLING_QUEUE_ABANDON",
}

normalised = []
for ev in events:
    raw_type = ev.get("event_type", "")
    mapped = type_map.get(raw_type, raw_type.upper())
    zone_name = ev.get("zone_name", "") or ""
    zone_id = ev.get("zone_id") or zone_name.replace(" ", "_").upper() or None
    track_id = ev.get("track_id", 0)
    visitor_id = ev.get("id_token") or f"VIS_{track_id:06x}"
    ts = (ev.get("event_timestamp") or ev.get("event_time")
          or ev.get("queue_join_ts") or "2026-04-10T12:00:00Z")

    normalised.append({
        "event_id": ev.get("queue_event_id") or str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": ev.get("camera_id", "CAM_ENTRY_01"),
        "visitor_id": visitor_id,
        "event_type": mapped,
        "timestamp": ts,
        "zone_id": zone_id,
        "dwell_ms": 0,
        "is_staff": ev.get("is_staff", False),
        "confidence": 0.90,
        "metadata": {
            "queue_depth": ev.get("queue_position_at_join"),
            "sku_zone": None,
            "session_seq": 1,
        },
    })

# 3. Ingest
r = requests.post(f"{BASE}/events/ingest", json={"events": normalised})
print("\nINGEST:", r.status_code)
print(json.dumps(r.json(), indent=2))

# 4. Metrics
r = requests.get(f"{BASE}/stores/STORE_BLR_002/metrics?date=2026-04-10")
print("\nMETRICS:", r.status_code)
print(json.dumps(r.json(), indent=2))

# 5. Funnel
r = requests.get(f"{BASE}/stores/STORE_BLR_002/funnel?date=2026-04-10")
print("\nFUNNEL:", r.status_code)
print(json.dumps(r.json(), indent=2))

# 6. Heatmap
r = requests.get(f"{BASE}/stores/STORE_BLR_002/heatmap?date=2026-04-10")
print("\nHEATMAP:", r.status_code)
print(json.dumps(r.json(), indent=2))
