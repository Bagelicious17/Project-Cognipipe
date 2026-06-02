"""
CogniPipe — FastAPI Application
==================================

Application factory with lifespan handler, CORS middleware,
and router mounting.

Usage::

    # Development
    python run.py

    # Production
    uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import pipeline
from app.routers.schemas import HealthResponse
from config import settings

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


# ──────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler.

    On startup:
    - Validate that GEMINI_API_KEY is set.
    - Log the configured model name and server settings.
    """
    # ── Startup ───────────────────────────────────────────────────
    try:
        _ = settings.gemini_api_key
        logger.info(
            "CogniPipe API v%s starting — model=%s, host=%s, port=%d",
            _VERSION,
            settings.gemini_model_name,
            settings.api_host,
            settings.api_port,
        )
    except ValueError as e:
        logger.error("Startup check failed: %s", e)
        raise

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("CogniPipe API shutting down.")


# ──────────────────────────────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Fully configured ``FastAPI`` instance with CORS,
        health check, and pipeline router mounted.
    """
    app = FastAPI(
        title="CogniPipe API",
        description=(
            "Upload a CSV dataset and receive a production-ready, "
            "AI-generated ML pipeline (Python script + Jupyter notebook)."
        ),
        version=_VERSION,
        lifespan=_lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────
    app.include_router(
        pipeline.router,
        prefix="/api/v1/pipeline",
        tags=["pipeline"],
    )

    # ── Health check ──────────────────────────────────────────────
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Liveness check",
    )
    async def health():
        """Return server status, version, and configured Gemini model."""
        return HealthResponse(
            status="healthy",
            version=_VERSION,
            gemini_model=settings.gemini_model_name,
        )

    return app
