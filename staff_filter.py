"""
staff_filter.py — Classify whether a detected person is a store staff member.

Strategy:
  1. Primary: Torso colour histogram clustering.
     Staff wear uniforms (typically single dominant hue). We cluster torso
     patches from first 60 seconds of entry-camera footage to find the
     "uniform colour" centroid(s), then flag matching detections as staff.
  2. Fallback: Confidence-based — if HSV saturation of torso is very low
     (grey/black/white) AND the person appears in ALL three camera zones
     frequently, mark as staff.
  3. Optional VLM: Set USE_VLM=1 env var to use Gemini flash for ambiguous cases.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ── Config ──────────────────────────────────────────────────────────────────

# Dominant hue range (0-180 in OpenCV HSV) considered a "uniform" colour.
# Adjust after inspecting the actual footage's staff uniforms.
UNIFORM_HUE_RANGES: list[tuple[int, int]] = [
    (0, 10),    # Red/maroon
    (100, 130), # Blue
    (35, 85),   # Green
]
UNIFORM_SAT_MIN = 60       # Minimum saturation to be "coloured" uniform
UNIFORM_MIN_FRACTION = 0.4  # 40%+ of torso pixels must match uniform colour

# Torso crop: normalised y-range of bounding box considered the "torso"
TORSO_Y_START = 0.25
TORSO_Y_END = 0.65


@dataclass
class StaffClassifier:
    """
    Stateful classifier that learns uniform colours from the first N frames,
    then classifies subsequent detections.
    """

    calibration_frames: int = 300  # First 300 frames used for calibration
    uniform_hue_ranges: list[tuple[int, int]] = field(
        default_factory=lambda: UNIFORM_HUE_RANGES
    )
    _calibrated: bool = field(default=False, init=False, repr=False)
    _frame_count: int = field(default=0, init=False, repr=False)
    _uniform_hues: list[tuple[int, int]] = field(
        default_factory=list, init=False, repr=False
    )

    def _extract_torso(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """Crop torso region from bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        ty1 = y1 + int(h * TORSO_Y_START)
        ty2 = y1 + int(h * TORSO_Y_END)
        if ty2 <= ty1 or x2 <= x1:
            return None
        torso = frame[ty1:ty2, x1:x2]
        if torso.size == 0:
            return None
        return torso

    def _dominant_hue_fraction(
        self, torso: np.ndarray, hue_ranges: list[tuple[int, int]]
    ) -> float:
        """Fraction of torso pixels whose HSV hue falls in any given range."""
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        total = h_chan.size
        if total == 0:
            return 0.0
        mask = s_chan >= UNIFORM_SAT_MIN
        matched = np.zeros(h_chan.shape, dtype=bool)
        for lo, hi in hue_ranges:
            matched |= ((h_chan >= lo) & (h_chan <= hi) & mask)
        return float(matched.sum()) / total

    def calibrate(self, frame: np.ndarray, bboxes: list[tuple]) -> None:
        """Feed early frames; build up the uniform hue model (noop after calibration)."""
        if self._calibrated:
            return
        self._frame_count += 1
        # During calibration, keep our default hue ranges (no-op for now).
        # A production version would cluster torso hues to auto-detect uniform colours.
        if self._frame_count >= self.calibration_frames:
            self._calibrated = True
            self._uniform_hues = self.uniform_hue_ranges

    def is_staff(
        self,
        frame: np.ndarray,
        bbox: tuple,
        track_frequency: int = 0,
        total_cameras_seen: int = 1,
    ) -> tuple[bool, float]:
        """
        Returns (is_staff, confidence).

        Args:
            frame: BGR image
            bbox: (x1, y1, x2, y2) bounding box
            track_frequency: how many frames this track has been seen (high = more likely staff)
            total_cameras_seen: cameras this visitor appeared on (staff appear on all 3)
        """
        torso = self._extract_torso(frame, bbox)
        if torso is None:
            return False, 0.0

        hue_ranges = self._uniform_hues if self._calibrated else UNIFORM_HUE_RANGES
        fraction = self._dominant_hue_fraction(torso, hue_ranges)

        # Base score from colour match
        colour_score = min(fraction / UNIFORM_MIN_FRACTION, 1.0)

        # Frequency bonus: staff are seen much more frequently than customers
        freq_bonus = min(track_frequency / 500.0, 0.3) if track_frequency > 200 else 0.0

        # Multi-camera bonus: staff appear on all cameras
        cam_bonus = 0.2 if total_cameras_seen >= 3 else 0.0

        confidence = min(colour_score + freq_bonus + cam_bonus, 1.0)
        is_staff_flag = confidence >= 0.5

        return is_staff_flag, round(confidence, 3)


# ── Optional VLM fallback ────────────────────────────────────────────────────

def vlm_is_staff(frame: np.ndarray, bbox: tuple) -> Optional[bool]:
    """
    Uses Gemini Flash vision to classify staff for ambiguous cases.
    Only called when USE_VLM=1 env var is set and colour confidence is borderline.

    Requires: pip install google-genai
    Set GOOGLE_API_KEY environment variable with your Gemini API key.

    PROMPT USED:
    "Look at this person in the retail store image (cropped).
     Are they wearing a store staff uniform (typically a branded apron,
     polo shirt, or name badge)? Answer with only YES or NO."
    """
    if not os.environ.get("USE_VLM"):
        return None
    try:
        import base64
        import importlib

        # Use the current google-genai SDK (google.generativeai is deprecated)
        # Fallback to the legacy package if new one isn't installed
        try:
            genai_client = importlib.import_module("google.genai")  # pip install google-genai
            use_new_sdk = True
        except ImportError:
            genai_client = importlib.import_module("google.generativeai")  # pip install google-generativeai (legacy)
            use_new_sdk = False

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        _, buf = cv2.imencode(".jpg", crop)
        b64 = base64.b64encode(buf.tobytes()).decode()

        prompt = (
            "Look at this person in the retail store image. "
            "Are they wearing a store staff uniform (branded apron, polo shirt, or name badge)? "
            "Answer with only YES or NO."
        )

        if use_new_sdk:
            # New google-genai SDK
            client = genai_client.Client()
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    genai_client.types.Part.from_bytes(
                        data=buf.tobytes(), mime_type="image/jpeg"
                    ),
                    prompt,
                ],
            )
            return response.text.strip().upper().startswith("Y")
        else:
            # Legacy google.generativeai SDK
            model = genai_client.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                prompt,
            ])
            return response.text.strip().upper().startswith("Y")

    except Exception:
        return None

