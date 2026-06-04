"""
middleware.py — Structured request logging middleware for FastAPI.

Logs every request with:
  trace_id, store_id (if path param), endpoint, latency_ms,
  event_count (for ingest), status_code

JSON format for easy ingestion by log aggregators.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = str(uuid.uuid4())[:8]
        start_ts = time.perf_counter()

        # Extract store_id from path if present
        path_parts = request.url.path.split("/")
        store_id = None
        if "stores" in path_parts:
            idx = path_parts.index("stores")
            if idx + 1 < len(path_parts):
                store_id = path_parts[idx + 1]

        # Forward request
        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)

        # Try to extract event_count for ingest endpoint
        event_count = None
        if "ingest" in request.url.path:
            event_count = request.state.__dict__.get("event_count")

        log_entry = {
            "trace_id": trace_id,
            "store_id": store_id,
            "endpoint": f"{request.method} {request.url.path}",
            "latency_ms": latency_ms,
            "event_count": event_count,
            "status_code": response.status_code,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(json.dumps(log_entry))

        # Attach trace_id to response headers for debugging
        response.headers["X-Trace-Id"] = trace_id
        return response
