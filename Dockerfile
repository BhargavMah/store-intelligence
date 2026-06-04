FROM python:3.11-slim

WORKDIR /app

# System dependencies (none needed for pure FastAPI + psycopg2-binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.api.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy shared schema + layout file
COPY schema.py ./
COPY store_layout.json ./

# Copy entire app directory
COPY app/ ./

# Health check so docker compose knows when we are ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with 2 workers (workers=2 is safe for SQLite too since we use check_same_thread=False)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
