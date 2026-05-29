# ──────────────────────────────────────────────────────────────────────
# CogniPipe — Multi-stage Production Dockerfile
# ──────────────────────────────────────────────────────────────────────
# Base:   python:3.11-slim (minimal footprint, ~120 MB base)
# Port:   8080 (Cloud Run default)
# User:   non-root 'appuser' for security
# ──────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
# so Cloud Run logging captures stdout/stderr immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System deps ──────────────────────────────────────────────────────
# gcc and python3-dev are needed by scipy / numpy wheel builds on slim.
# We clean up the apt cache afterwards to keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ──────────────────────────────────────
# Copy requirements.txt FIRST so Docker layer caching only reinstalls
# when dependencies actually change, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ──────────────────────────────────────────────
COPY . .

# ── Non-root user ───────────────────────────────────────────────────
# Security best practice: never run the app as root.
RUN adduser --disabled-password --gecos "" --uid 1001 appuser \
    && chown -R appuser:appuser /app

USER appuser

# ── Runtime ──────────────────────────────────────────────────────────
# Cloud Run injects PORT=8080, but we also EXPOSE it for documentation.
EXPOSE 8080

# --workers 1: Cloud Run scales horizontally via instances, not workers.
# --factory:   uses the create_app() factory pattern from app/main.py.
# No --reload: this is production.
CMD ["uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
