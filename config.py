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
        """Gemini model name (defaults to gemini-1.5-pro)."""
        return os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")


settings = Settings()
