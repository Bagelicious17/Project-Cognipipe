"""
CogniPipe — Application Configuration
=======================================
Loads settings from environment variables (via ``.env`` file).

Usage::

    from config import settings
    print(settings.gemini_api_key)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


class Settings:
    """Typed accessor for environment variables.

    All values are read once at import time.  If a required variable
    is missing, accessing its property will raise ``ValueError`` with
    a helpful message.
    """

    @property
    def gemini_api_key(self) -> str:
        """Google Gemini API key (required for Layer 2)."""
        key = os.getenv("GEMINI_API_KEY", "")
        if not key or key == "your-api-key-here":
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and paste your key:\n"
                "  cp .env.example .env"
            )
        return key

    @property
    def gemini_model_name(self) -> str:
        """Gemini model name (defaults to gemini-2.5-flash)."""
        return os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    # ── FastAPI settings ──────────────────────────────────────────

    @property
    def api_host(self) -> str:
        """Host to bind the API server to."""
        return os.getenv("API_HOST", "0.0.0.0")

    @property
    def api_port(self) -> int:
        """Port to bind the API server to."""
        return int(os.getenv("API_PORT", "8000"))

    @property
    def max_upload_mb(self) -> int:
        """Maximum upload file size in megabytes."""
        return int(os.getenv("MAX_UPLOAD_MB", "50"))

    @property
    def cors_origins(self) -> list[str]:
        """Allowed CORS origins (comma-separated in env)."""
        raw = os.getenv("CORS_ORIGINS", "*")
        return [o.strip() for o in raw.split(",")]

    @property
    def gemini_timeout_seconds(self) -> int:
        """Timeout for Gemini API calls in seconds."""
        return int(os.getenv("GEMINI_TIMEOUT_SECONDS", "300"))


settings = Settings()
