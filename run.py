"""
CogniPipe — Development Server Entry Point
=============================================

Start the API server with auto-reload for development::

    python run.py

For production, use uvicorn directly::

    uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8080
"""

import logging
import os

import uvicorn

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)

if __name__ == "__main__":
    # Cloud Run injects PORT env var automatically (default 8080).
    # This lets the same run.py work locally and inside the container.
    port = int(os.getenv("PORT", settings.api_port))

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.api_host,
        port=port,
        reload=True,
    )
