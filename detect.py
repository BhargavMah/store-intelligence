"""
detect.py — Main detection + tracking script.

Usage:
    python detect.py \\
        --video   clips/Store1/CAM_ENTRY.mp4 \\
        --store   STORE_BLR_002 \\
        --camera  CAM_ENTRY_01 \\
        --start   "2026-04-10T12:00:00" \\
        --out     data/events.jsonl

    # Or process all clips in a directory:
    python detect.py --store-dir clips/Store1/ --store STORE_BLR_001 --out data/events.jsonl

Pipeline per clip:
  1. Read frames from video file (OpenCV)
  2. Run YOLOv8 inference every FRAME_SKIP frames
  3. Pass detections to ByteTrack (via ultralytics)
  4. Extract lightweight colour embedding per track (HSV histogram)
  5. Run Tracker.update() → get events
  6. Validate + write events via EventEmitter
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows terminal encoding (CP1252 can't print unicode arrows etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7

import cv2

# ── Lazy imports (so the file can be imported in tests without GPU) ─────────

def _import_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)


# ── Config ───────────────────────────────────────────────────────────────────

FRAME_SKIP = 3          # Process every Nth frame (15fps / 3 = 5 fps effective)
CONFIDENCE_THRESHOLD = 0.35
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")  # nano by default; use yolov8s for better accuracy
PERSON_CLASS_ID = 0     # COCO class 0 = person


# ── Colour embedding (lightweight Re-ID fallback) ────────────────────────────

def _extract_colour_embedding(frame: "cv2.Mat", x1: int, y1: int, x2: int, y2: int) -> "np.ndarray":
    """
    48-dimensional HSV colour histogram over the torso region of a bounding box.
    Used as a fast Re-ID embedding when a dedicated Re-ID model is unavailable.
    """
    import numpy as np

    h_total = y2 - y1
    ty1 = y1 + int(h_total * 0.25)
    ty2 = y1 + int(h_total * 0.65)
    crop = frame[ty1:ty2, x1:x2]
    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    norm = feat.sum()
    return feat / norm if norm > 0 else feat


# ── Billing queue depth estimator ────────────────────────────────────────────

def _estimate_queue_depth(tracks: list[dict], billing_zone_id: str, zone_mapper) -> int:
    """Count how many active tracks are currently in the billing zone."""
    count = 0
    for t in tracks:
        cx = ((t["x1"] + t["x2"]) / 2) / t.get("frame_width", 1920)
        cy = ((t["y1"] + t["y2"]) / 2) / t.get("frame_height", 1080)
        zid = zone_mapper.get_zone(t.get("camera_id", ""), cx, cy)
        if zid == billing_zone_id:
            count += 1
    return max(0, count - 1)  # subtract 1 for the currently-being-served person


# ── Main processing function ─────────────────────────────────────────────────

def process_clip(
    video_path: str,
    store_id: str,
    camera_id: str,
    clip_start_dt: datetime,
    emitter,
    tracker,
    zone_mapper,
    staff_classifier,
) -> dict:
    """Process a single video clip and emit events. Returns summary stats."""
    import numpy as np

    YOLO = _import_yolo()
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[detect] {Path(video_path).name} | {total_frames} frames @ {fps:.1f}fps | {frame_w}x{frame_h}")

    frame_idx = 0
    processed = 0
    total_events = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # ── YOLO inference ───────────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        tracks = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                tid_tensor = boxes.id
                if tid_tensor is None:
                    continue
                tid = int(tid_tensor[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                embedding = _extract_colour_embedding(frame, x1, y1, x2, y2)

                tracks.append({
                    "track_id": tid,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf,
                    "embedding": embedding,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                    "camera_id": camera_id,
                })

        # ── Staff calibration ────────────────────────────────────────────
        staff_classifier.calibrate(frame, [(t["x1"], t["y1"], t["x2"], t["y2"]) for t in tracks])

        # ── Queue depth (for billing cameras) ────────────────────────────
        queue_depth = 0
        cam_type = zone_mapper.get_camera_type(camera_id)
        if cam_type == "billing":
            queue_depth = _estimate_queue_depth(tracks, "BILLING_COUNTER", zone_mapper)

        # ── Tracker update → events ──────────────────────────────────────
        raw_events = tracker.update(
            frame_idx=frame_idx,
            fps=fps,
            camera_id=camera_id,
            tracks=tracks,
            zone_mapper=zone_mapper,
            staff_classifier=staff_classifier,
            frame=frame,
            queue_depth=queue_depth,
        )

        emitted = emitter.emit_batch(raw_events)
        total_events += len(emitted)
        processed += 1
        frame_idx += 1

        if processed % 100 == 0:
            pct = (frame_idx / max(total_frames, 1)) * 100
            print(f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames} | events so far: {total_events}")

    cap.release()
    print(f"[detect] Done. {processed} frames processed, {total_events} events emitted.")
    return {"frames_processed": processed, "events_emitted": total_events}


# ── CLI entrypoint ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Purplle Store Intelligence — Detection Pipeline")
    p.add_argument("--video", help="Path to a single video clip")
    p.add_argument("--store-dir", help="Directory containing multiple clips for a store")
    p.add_argument("--store", required=True, help="Store ID (e.g. STORE_BLR_002)")
    p.add_argument("--camera", default="CAM_ENTRY_01", help="Camera ID for the clip")
    p.add_argument("--start", default="2026-04-10T12:00:00", help="ISO-8601 start time for clip")
    p.add_argument("--out", default="data/events.jsonl", help="Output JSONL path")
    p.add_argument("--api-url", help="If set, POST events to this API URL instead of/in addition to file")
    return p.parse_args()


def _camera_id_from_filename(filename: str) -> str:
    """Infer camera ID from clip filename."""
    name = Path(filename).stem.lower()
    if "entry" in name:
        return "CAM_ENTRY_01"
    if "billing" in name or "bill" in name:
        return "CAM_BILLING_01"
    return "CAM_FLOOR_01"


def main():
    args = _parse_args()

    from emit import EventEmitter
    from staff_filter import StaffClassifier
    from tracker import Tracker
    from zone_mapper import ZoneMapper

    clip_start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    tracker = Tracker(store_id=args.store, clip_start_dt=clip_start)
    zone_mapper = ZoneMapper(store_id=args.store)
    staff_classifier = StaffClassifier()

    clips = []
    if args.video:
        clips.append((args.video, args.camera))
    elif args.store_dir:
        store_dir = Path(args.store_dir)
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for vf in sorted(store_dir.glob(ext)):
                cam_id = _camera_id_from_filename(vf.name)
                clips.append((str(vf), cam_id))

    if not clips:
        print("[ERROR] No video clips found. Use --video or --store-dir.", file=sys.stderr)
        sys.exit(1)

    print(f"[detect] Processing {len(clips)} clip(s) -> {args.out}")

    with EventEmitter(output_path=args.out) as emitter:
        for video_path, camera_id in clips:
            print(f"\n[detect] Processing: {video_path} (camera: {camera_id})")
            try:
                process_clip(
                    video_path=video_path,
                    store_id=args.store,
                    camera_id=camera_id,
                    clip_start_dt=clip_start,
                    emitter=emitter,
                    tracker=tracker,
                    zone_mapper=zone_mapper,
                    staff_classifier=staff_classifier,
                )
            except Exception as exc:
                print(f"[ERROR] Failed to process {video_path}: {exc}", file=sys.stderr)

        print(f"\n[detect] Final stats: {emitter.stats}")


if __name__ == "__main__":
    main()
