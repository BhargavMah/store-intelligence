"""
tracker.py — ByteTrack-based multi-person tracker with Re-ID and entry/exit logic.

Responsibilities:
  - Wrap the Ultralytics ByteTrack results into a clean per-track state machine
  - Determine entry/exit direction using the virtual crossing line
  - Maintain Re-ID embedding history for cross-camera deduplication
  - Detect re-entry: same embedding returning after a prior EXIT
  - Manage per-visitor session state
"""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

REID_COSINE_THRESHOLD = 0.75   # Similarity above which two tracks = same person
REENTRY_TIMEOUT_S = 300        # After 5 min outside, treat re-appearance as new visit
SESSION_EXPIRE_S = 600         # Sessions idle > 10 min are force-closed
DWELL_EMIT_INTERVAL_S = 30     # Emit ZONE_DWELL every 30 seconds of continuous dwell


class TrackState(str, Enum):
    OUTSIDE = "outside"       # Not yet entered, or exited
    INSIDE = "inside"         # Currently inside the store
    BILLING = "billing"       # Currently in billing queue


@dataclass
class TrackHistory:
    """Per-track sliding window of normalised centroids for direction detection."""
    centroids: list[tuple[float, float]] = field(default_factory=list)
    max_len: int = 30

    def add(self, cx: float, cy: float) -> None:
        self.centroids.append((cx, cy))
        if len(self.centroids) > self.max_len:
            self.centroids.pop(0)

    def velocity_y(self) -> float:
        """Average Y velocity (positive = moving down the frame)."""
        if len(self.centroids) < 5:
            return 0.0
        ys = [c[1] for c in self.centroids[-10:]]
        return (ys[-1] - ys[0]) / max(len(ys) - 1, 1)


@dataclass
class VisitorSession:
    visitor_id: str
    track_id: int
    store_id: str
    camera_id: str
    state: TrackState = TrackState.OUTSIDE
    entry_time: Optional[float] = None       # unix timestamp
    last_seen: float = field(default_factory=time.time)
    current_zone: Optional[str] = None
    zone_entry_time: Optional[float] = None
    last_dwell_emit: Optional[float] = None
    session_seq: int = 0
    cameras_seen: set = field(default_factory=set)
    embedding: Optional[np.ndarray] = None   # Re-ID feature vector
    history: TrackHistory = field(default_factory=TrackHistory)
    total_frames: int = 0

    def is_stale(self) -> bool:
        return (time.time() - self.last_seen) > SESSION_EXPIRE_S


class ReIDStore:
    """
    Maintains a library of (visitor_id → embedding) for cross-camera re-identification.
    """

    def __init__(self) -> None:
        self._library: dict[str, np.ndarray] = {}         # visitor_id → embedding
        self._exit_times: dict[str, float] = {}            # visitor_id → unix exit time

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def find_match(
        self, embedding: np.ndarray, exclude_ids: set[str] | None = None
    ) -> tuple[Optional[str], float]:
        """Return (visitor_id, similarity) of best match, or (None, 0)."""
        best_id, best_sim = None, 0.0
        for vid, emb in self._library.items():
            if exclude_ids and vid in exclude_ids:
                continue
            sim = self._cosine_similarity(embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = vid
        return best_id, best_sim

    def register(self, visitor_id: str, embedding: np.ndarray) -> None:
        self._library[visitor_id] = embedding

    def record_exit(self, visitor_id: str) -> None:
        self._exit_times[visitor_id] = time.time()

    def seconds_since_exit(self, visitor_id: str) -> float:
        t = self._exit_times.get(visitor_id)
        return (time.time() - t) if t else float("inf")


def make_visitor_id(seed: str) -> str:
    """Generate a short deterministic visitor token from a seed string."""
    h = hashlib.md5(seed.encode()).hexdigest()[:6]
    return f"VIS_{h}"


class Tracker:
    """
    Wraps per-camera tracking results and maintains the global visitor session table.
    """

    def __init__(self, store_id: str, clip_start_dt: datetime) -> None:
        self.store_id = store_id
        self.clip_start_dt = clip_start_dt
        self._sessions: dict[int, VisitorSession] = {}   # track_id → session
        self._reid_store = ReIDStore()
        self._visitor_counter = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def frame_timestamp(self, frame_idx: int, fps: float) -> str:
        """Convert frame index to ISO-8601 UTC timestamp."""
        offset_s = frame_idx / fps
        dt = self.clip_start_dt.replace(tzinfo=timezone.utc)
        from datetime import timedelta
        ts = dt + timedelta(seconds=offset_s)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def update(
        self,
        frame_idx: int,
        fps: float,
        camera_id: str,
        tracks: list[dict],
        zone_mapper,
        staff_classifier,
        frame: np.ndarray,
        queue_depth: int = 0,
    ) -> list[dict]:
        """
        Process one frame of tracking results.

        Args:
            tracks: list of dicts with keys:
                track_id, x1, y1, x2, y2, confidence, embedding (optional)
        Returns:
            list of event dicts ready for emit.py
        """
        timestamp = self.frame_timestamp(frame_idx, fps)
        events = []
        cam_type = zone_mapper.get_camera_type(camera_id)
        entry_line_y = zone_mapper.get_entry_line_y(camera_id)

        active_track_ids = {t["track_id"] for t in tracks}

        # Track visitor_ids assigned in THIS frame to prevent group cross-matching
        frame_assigned_ids: set[str] = {
            s.visitor_id for s in self._sessions.values()
        }

        for track in tracks:
            tid = track["track_id"]
            x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
            conf = track.get("confidence", 0.9)
            embedding = track.get("embedding")

            # Normalised centroid
            cx = ((x1 + x2) / 2) / track.get("frame_width", 1920)
            cy = ((y1 + y2) / 2) / track.get("frame_height", 1080)

            # ── Classify staff ──────────────────────────────────────────
            session = self._sessions.get(tid)
            frames_seen = session.total_frames if session else 0
            cams_seen = len(session.cameras_seen) if session else 1
            is_staff, staff_conf = staff_classifier.is_staff(
                frame, (x1, y1, x2, y2),
                track_frequency=frames_seen,
                total_cameras_seen=cams_seen,
            )

            # ── Re-ID / visitor assignment ──────────────────────────────
            if tid not in self._sessions:
                visitor_id, is_reentry = self._resolve_visitor(
                    embedding, camera_id, exclude_active_ids=frame_assigned_ids
                )
                session = VisitorSession(
                    visitor_id=visitor_id,
                    track_id=tid,
                    store_id=self.store_id,
                    camera_id=camera_id,
                    embedding=embedding,
                )
                self._sessions[tid] = session
                frame_assigned_ids.add(visitor_id)  # prevent next track matching this one

                # Entry event (if entry camera)
                if cam_type == "entry":
                    ev_type = "REENTRY" if is_reentry else "ENTRY"
                    session.state = TrackState.INSIDE
                    session.entry_time = time.time()
                    session.session_seq += 1
                    events.append(self._make_event(
                        session, ev_type, timestamp, None, 0, conf,
                        is_staff=is_staff
                    ))

            session = self._sessions[tid]
            session.last_seen = time.time()
            session.history.add(cx, cy)
            session.cameras_seen.add(camera_id)
            session.total_frames += 1
            if embedding is not None:
                session.embedding = embedding
                self._reid_store.register(session.visitor_id, embedding)

            # ── Zone detection ──────────────────────────────────────────
            zone_id = zone_mapper.get_zone(camera_id, cx, cy)
            now = time.time()

            if zone_id != session.current_zone:
                # Zone changed
                if session.current_zone is not None:
                    # ZONE_EXIT from previous zone
                    dwell = int((now - (session.zone_entry_time or now)) * 1000)
                    session.session_seq += 1
                    events.append(self._make_event(
                        session, "ZONE_EXIT", timestamp,
                        session.current_zone, dwell, conf, is_staff=is_staff
                    ))

                if zone_id is not None:
                    # ZONE_ENTER new zone
                    session.current_zone = zone_id
                    session.zone_entry_time = now
                    session.last_dwell_emit = now
                    session.session_seq += 1

                    zone_meta = zone_mapper.get_zone_meta(zone_id)
                    ev_type = "ZONE_ENTER"

                    # Billing-specific logic
                    if zone_meta.get("zone_type") == "BILLING":
                        session.state = TrackState.BILLING
                        if queue_depth > 0:
                            ev_type = "BILLING_QUEUE_JOIN"

                    events.append(self._make_event(
                        session, ev_type, timestamp, zone_id, 0, conf,
                        is_staff=is_staff, queue_depth=queue_depth if ev_type == "BILLING_QUEUE_JOIN" else None
                    ))
                else:
                    session.current_zone = None
                    session.zone_entry_time = None
                    session.last_dwell_emit = None

            # ── Dwell event ─────────────────────────────────────────────
            if (
                zone_id is not None
                and session.zone_entry_time is not None
                and session.last_dwell_emit is not None
            ):
                dwell_so_far = now - session.zone_entry_time
                time_since_last = now - session.last_dwell_emit
                if dwell_so_far >= DWELL_EMIT_INTERVAL_S and time_since_last >= DWELL_EMIT_INTERVAL_S:
                    session.last_dwell_emit = now
                    session.session_seq += 1
                    events.append(self._make_event(
                        session, "ZONE_DWELL", timestamp,
                        zone_id, int(dwell_so_far * 1000), conf, is_staff=is_staff
                    ))

        # ── Handle disappeared tracks (EXIT) ────────────────────────────
        for tid, session in list(self._sessions.items()):
            if tid not in active_track_ids and cam_type == "entry":
                if session.state == TrackState.INSIDE:
                    vel_y = session.history.velocity_y()
                    # Moving towards exit (downward in typical CCTV = outbound)
                    if vel_y > 0.01 or session.is_stale():
                        session.state = TrackState.OUTSIDE
                        session.session_seq += 1
                        events.append(self._make_event(
                            session, "EXIT", self.frame_timestamp(frame_idx, fps),
                            None, 0, 0.85,
                        ))
                        self._reid_store.record_exit(session.visitor_id)
                        # Keep session for re-entry detection, don't pop

                    # Billing queue abandon check
                    if session.state == TrackState.BILLING:
                        session.state = TrackState.INSIDE
                        session.session_seq += 1
                        events.append(self._make_event(
                            session, "BILLING_QUEUE_ABANDON", timestamp,
                            session.current_zone, 0, 0.8
                        ))

        return events

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _resolve_visitor(
        self, embedding: Optional[np.ndarray], camera_id: str,
        exclude_active_ids: set[str] | None = None,
    ) -> tuple[str, bool]:
        """Return (visitor_id, is_reentry).

        Args:
            exclude_active_ids: visitor_ids already assigned to tracks in this
                frame — prevents group-entry tracks from being matched to each other.
        """
        if embedding is not None:
            matched_id, sim = self._reid_store.find_match(
                embedding, exclude_ids=exclude_active_ids
            )
            if matched_id and sim >= REID_COSINE_THRESHOLD:
                secs = self._reid_store.seconds_since_exit(matched_id)
                is_reentry = secs < REENTRY_TIMEOUT_S
                return matched_id, is_reentry

        # New visitor
        self._visitor_counter += 1
        seed = f"{self.store_id}_{camera_id}_{self._visitor_counter}_{time.time()}"
        return make_visitor_id(seed), False

    def _make_event(
        self,
        session: VisitorSession,
        event_type: str,
        timestamp: str,
        zone_id: Optional[str],
        dwell_ms: int,
        confidence: float,
        is_staff: bool = False,
        queue_depth: Optional[int] = None,
    ) -> dict:
        zone_meta = {}
        if zone_id:
            zone_meta = {
                "zone_id": zone_id,
                "zone_meta": session.store_id,
            }
        return {
            "store_id": session.store_id,
            "camera_id": session.camera_id,
            "visitor_id": session.visitor_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": round(min(max(confidence, 0.0), 1.0), 3),
            "metadata": {
                "queue_depth": queue_depth,
                "sku_zone": None,
                "session_seq": session.session_seq,
            },
        }
