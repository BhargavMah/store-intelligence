"""
zone_mapper.py — Maps (camera_id, normalised centroid x/y) → zone_id

Given a bounding box centroid from a particular camera, this module determines
which named zone (if any) the person is currently occupying, using the polygon
definitions in store_layout.json.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import numpy as np


LAYOUT_PATH = os.path.join(os.path.dirname(__file__), "store_layout.json")


@lru_cache(maxsize=None)
def _load_layout() -> dict:
    with open(LAYOUT_PATH, "r") as f:
        return json.load(f)


def _point_in_polygon(px: float, py: float, polygon: list[list[float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test (normalised coords)."""
    n = len(polygon)
    inside = False
    x, y = px, py
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class ZoneMapper:
    """
    Maps a normalised centroid (cx, cy) within a camera frame to a zone_id.

    Coordinates are normalised 0..1 relative to frame width/height.
    """

    def __init__(self, store_id: str) -> None:
        layout = _load_layout()
        if store_id not in layout["stores"]:
            # Fall back to first store if unknown
            store_id = next(iter(layout["stores"]))
        self.store_data = layout["stores"][store_id]
        # Build zone lookup keyed by camera
        self._camera_zones: dict[str, list[dict]] = {}
        for zone in self.store_data["zones"]:
            cam = zone["camera_id"]
            self._camera_zones.setdefault(cam, []).append(zone)

    def get_zone(self, camera_id: str, cx: float, cy: float) -> Optional[str]:
        """Return zone_id for centroid (cx, cy) in given camera, or None."""
        for zone in self._camera_zones.get(camera_id, []):
            poly = zone["polygon_norm"]
            if _point_in_polygon(cx, cy, poly):
                return zone["zone_id"]
        return None

    def get_zone_meta(self, zone_id: str) -> dict:
        """Return zone metadata dict for a zone_id."""
        for zone in self.store_data["zones"]:
            if zone["zone_id"] == zone_id:
                return zone
        return {}

    def get_camera_type(self, camera_id: str) -> str:
        """Return 'entry', 'floor', or 'billing'."""
        cameras = self.store_data.get("cameras", {})
        return cameras.get(camera_id, {}).get("type", "floor")

    def get_entry_line_y(self, camera_id: str) -> float:
        """Return normalised Y threshold for entry/exit detection."""
        cameras = self.store_data.get("cameras", {})
        return cameras.get(camera_id, {}).get("entry_line", {}).get("y", 0.65)

    def list_zones(self) -> list[str]:
        return [z["zone_id"] for z in self.store_data["zones"]]
