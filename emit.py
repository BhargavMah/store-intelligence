"""
emit.py — Event schema validation and JSONL emitter.

Wraps the raw event dicts from tracker.py into validated StoreEvent
Pydantic objects and writes them to a JSONL output file (or sends to
the API in streaming mode).
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Optional, TextIO

from schema import EventMetadata, EventType, StoreEvent


class EventEmitter:
    """
    Converts raw event dicts into validated StoreEvent objects and writes
    them to a JSONL file.  Also maintains an in-memory buffer for streaming.
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        stream: Optional[TextIO] = None,
    ) -> None:
        self._out: Optional[TextIO] = None
        self._stream = stream or sys.stdout
        self._rejected = 0
        self._accepted = 0

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._out = open(p, "a", encoding="utf-8")

    def emit(self, raw: dict) -> Optional[StoreEvent]:
        """Validate and emit one event.  Returns the StoreEvent or None on error."""
        try:
            # Normalise event_type to uppercase enum value
            raw["event_type"] = raw.get("event_type", "").upper()
            # Generate event_id if missing
            raw.setdefault("event_id", str(uuid.uuid4()))
            # Coerce metadata
            meta = raw.pop("metadata", {}) or {}
            raw["metadata"] = EventMetadata(**meta)

            event = StoreEvent(**raw)
            self._accepted += 1
            line = event.model_dump_json()

            if self._out:
                self._out.write(line + "\n")
                self._out.flush()

            return event

        except Exception as exc:
            self._rejected += 1
            # Log but don't crash — low-confidence events are still emitted
            print(f"[EMIT WARN] rejected event: {exc} | raw={raw}", file=sys.stderr)
            return None

    def emit_batch(self, raws: list[dict]) -> list[StoreEvent]:
        """Emit a list of raw event dicts; return successfully validated events."""
        return [e for raw in raws if (e := self.emit(raw)) is not None]

    @property
    def stats(self) -> dict:
        return {"accepted": self._accepted, "rejected": self._rejected}

    def close(self) -> None:
        if self._out:
            self._out.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
