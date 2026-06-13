"""
CogniPipe — Pipeline Router
===============================

API endpoints for CSV upload, pipeline generation, profiling, and
file downloads.

Endpoints:
    POST /generate       — Full pipeline: CSV -> .py + .ipynb + requirements.txt
    POST /profile-only   — Layer 1 only: CSV -> ProfileResult JSON
    POST /download/script       — Download generated .py as a file
    POST /download/notebook     — Download generated .ipynb as a file
    POST /download/requirements — Download generated requirements.txt as a file
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from functools import partial

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.routers.schemas import (
    DownloadRequest,
    ErrorResponse,
    PipelineResponse,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamProgressEvent,
)
from config import settings
from models.schemas import ProfileResult
from services.code_assembler import CodeAssembler
from services.data_profiler import DataProfiler
from services.gemini_orchestrator import GeminiOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

_MAX_BYTES = settings.max_upload_mb * 1024 * 1024


def _sanitize_error(exc: Exception) -> str:
    """Convert raw exceptions into clean, user-facing error messages.

    Maps known Gemini API error codes and common failure patterns to
    friendly messages. Never leaks internal details to the frontend.
    """
    raw = str(exc).lower()

    if "429" in raw or "resource_exhausted" in raw or "quota" in raw:
        return (
            "The AI service has reached its usage limit for today. "
            "Please try again later or contact the administrator."
        )
    if "403" in raw or "permission_denied" in raw:
        return (
            "The AI service is not authorized. "
            "Please check the API key configuration."
        )
    if "400" in raw or "invalid_argument" in raw:
        return (
            "The AI service received an invalid request. "
            "Please try uploading a different dataset."
        )
    if "404" in raw or "not_found" in raw:
        return (
            "The configured AI model was not found. "
            "Please contact the administrator."
        )
    if "timeout" in raw or "deadline" in raw:
        return (
            "The AI service took too long to respond. "
            "Please try again with a smaller dataset."
        )
    if "connection" in raw or "network" in raw or "dns" in raw:
        return (
            "Unable to reach the AI service. "
            "Please check your internet connection and try again."
        )

    # Fallback — generic but clean
    return (
        "An unexpected error occurred during AI analysis. "
        "Please try again. If the issue persists, contact support."
    )

async def _validate_and_read_csv(file: UploadFile) -> pd.DataFrame:
    """Validate an uploaded file and return a pandas DataFrame.

    Validation order (per Addition 2):
    1. Extension must be .csv or .xlsx → 400
    2. Size ≤ MAX_UPLOAD_MB → 413
    3. File must not be empty → 400
    4. Must be parseable by pandas → 400
    5. At least 10 rows and 2 columns → 400

    Raises:
        HTTPException: With specific status code and message for each
            validation failure.
    """
    # 1. Extension check
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("csv", "xlsx"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "detail": (
                    f"Unsupported file extension '.{ext}'. "
                    "Only .csv and .xlsx files are accepted."
                ),
                "stage": "upload",
            },
        )

    # 2. Size check
    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "detail": (
                    f"File size ({len(contents) / (1024 * 1024):.1f} MB) "
                    f"exceeds the {settings.max_upload_mb} MB limit. "
                    "Try reducing the dataset or increasing MAX_UPLOAD_MB."
                ),
                "stage": "upload",
            },
        )

    # 3. Empty check
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "detail": "Uploaded file is empty (0 bytes).",
                "stage": "upload",
            },
        )

    # 4. Parseability check
    try:
        if ext == "xlsx":
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        else:
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "detail": (
                    f"Failed to parse file as {ext.upper()}: {e}. "
                    "Ensure the file is a valid, well-formed dataset."
                ),
                "stage": "upload",
            },
        )

    # 5. Minimum dimensions check
    if df.shape[0] < 10 or df.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "detail": (
                    f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
                    "A minimum of 10 rows and 2 columns is required "
                    "for meaningful analysis."
                ),
                "stage": "upload",
            },
        )

    return df


# ──────────────────────────────────────────────────────────────────────
# POST /profile-only
# ──────────────────────────────────────────────────────────────────────

@router.post(
    "/profile-only",
    response_model=ProfileResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
    summary="Profile a dataset (Layer 1 only)",
    description=(
        "Upload a CSV or XLSX file and receive the DataProfiler output. "
        "Useful for inspecting column types, missing patterns, and "
        "dataset flags before committing to full pipeline generation."
    ),
)
async def profile_only(file: UploadFile):
    """Run Layer 1 profiling only."""
    df = await _validate_and_read_csv(file)

    try:
        loop = asyncio.get_event_loop()
        profiler = DataProfiler()
        profile = await asyncio.wait_for(
            loop.run_in_executor(None, partial(profiler.profile, df)),
            timeout=settings.gemini_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "timeout",
                "detail": (
                    f"Profiling timed out after {settings.gemini_timeout_seconds}s. "
                    "Try uploading a smaller dataset or increasing "
                    "GEMINI_TIMEOUT_SECONDS."
                ),
                "stage": "profiling",
            },
        )
    except Exception as e:
        logger.exception("Profiling failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "profiling_error",
                "detail": (
                    "Data profiling failed unexpectedly. "
                    "Please try uploading a different dataset."
                ),
                "stage": "profiling",
            },
        )

    return profile


# ──────────────────────────────────────────────────────────────────────
# POST /generate  (NDJSON streaming)
# ──────────────────────────────────────────────────────────────────────
#
# NOTE: This endpoint returns an NDJSON stream (application/x-ndjson),
# not a single JSON object.  FastAPI's Swagger UI (/docs) cannot render
# streaming responses, so manual testing should use the React frontend
# or curl.  The /docs route is intentionally kept for other endpoints.
# ──────────────────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_class=StreamingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
    summary="Generate a full ML pipeline (streaming)",
    description=(
        "Upload a CSV or XLSX file and receive a streaming NDJSON response "
        "with real-time progress events followed by the complete pipeline. "
        "Events: {type:'progress'}, {type:'done'}, {type:'error'}."
    ),
)
async def generate_pipeline(file: UploadFile):
    """Run the full 3-layer pipeline with streaming progress.

    Returns an NDJSON stream of events:
    - ``progress``: intermediate progress updates (0–100%)
    - ``done``: final event with the complete PipelineResponse
    - ``error``: sent if any stage fails; stream ends after this
    """
    # Validate and parse the uploaded file *before* entering the stream.
    # This allows validation errors to be raised as normal HTTPExceptions.
    df = await _validate_and_read_csv(file)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _emit(event: dict) -> None:
        """Thread-safe: push an event dict onto the async queue."""
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _worker() -> None:
        """Synchronous worker that runs all 3 layers in a thread."""
        try:
            # ── Layer 1 — Profiling ─────────────────────────────────
            _emit(StreamProgressEvent(
                progress=5, message="Validating and parsing dataset…"
            ).model_dump())

            profiler = DataProfiler()
            profile = profiler.profile(df)

            _emit(StreamProgressEvent(
                progress=10, message="Data profiled — starting AI analysis…"
            ).model_dump())

            # ── Layer 2 — Gemini orchestration ──────────────────────
            def progress_cb(pct: int, msg: str) -> None:
                _emit(StreamProgressEvent(
                    progress=pct, message=msg
                ).model_dump())

            orchestrator = GeminiOrchestrator(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model_name,
            )
            gemini_result = orchestrator.run(
                profile, progress_callback=progress_cb
            )

            # ── Layer 3 — Code assembly ─────────────────────────────
            _emit(StreamProgressEvent(
                progress=90, message="Assembling pipeline code and notebook…"
            ).model_dump())

            assembler = CodeAssembler()
            pipeline = assembler.build(profile, gemini_result)

            response = PipelineResponse(
                python_script=pipeline.python_script,
                notebook_json=pipeline.notebook_json,
                requirements_txt=pipeline.requirements_txt,
                pipeline_summary=pipeline.pipeline_summary,
                generated_at=pipeline.generated_at,
                profiling_duration_seconds=profile.profiling_duration_seconds,
                orchestration_duration_seconds=(
                    gemini_result.orchestration_duration_seconds
                ),
            )

            # Explicit 100% before done
            _emit(StreamProgressEvent(
                progress=100, message="Pipeline ready!"
            ).model_dump())

            _emit(StreamDoneEvent(data=response).model_dump())

        except Exception as exc:
            logger.exception("Pipeline generation failed in worker")

            # Determine stage from exception context
            stage = "orchestration"
            if "profil" in type(exc).__name__.lower():
                stage = "profiling"

            _emit(StreamErrorEvent(
                error="generation_error",
                detail=_sanitize_error(exc),
                stage=stage,
            ).model_dump())

    # Launch the worker in a background thread
    loop.run_in_executor(None, _worker)

    async def _event_generator():
        """Async generator yielding NDJSON lines from the queue."""
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, default=str) + "\n"
                if event.get("type") in ("done", "error"):
                    # 0.5s delay after 100% so the UI can show it
                    if event.get("type") == "done":
                        await asyncio.sleep(0.5)
                    break
        except asyncio.CancelledError:
            # Client disconnected — clean up gracefully
            logger.info("Client disconnected during streaming")
            return

    return StreamingResponse(
        _event_generator(),
        media_type="application/x-ndjson",
    )


# ──────────────────────────────────────────────────────────────────────
# POST /download/*
# ──────────────────────────────────────────────────────────────────────

@router.post(
    "/download/script",
    summary="Download generated Python script",
    description="Returns the pipeline.py content as a downloadable file.",
    response_class=Response,
)
async def download_script(body: DownloadRequest):
    """Return the generated Python script as a file download."""
    return Response(
        content=body.content,
        media_type="text/x-python",
        headers={
            "Content-Disposition": 'attachment; filename="pipeline.py"',
        },
    )


@router.post(
    "/download/notebook",
    summary="Download generated Jupyter notebook",
    description="Returns the pipeline.ipynb content as a downloadable file.",
    response_class=Response,
)
async def download_notebook(body: DownloadRequest):
    """Return the generated Jupyter notebook as a file download."""
    return Response(
        content=body.content,
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": 'attachment; filename="pipeline.ipynb"',
        },
    )


@router.post(
    "/download/requirements",
    summary="Download generated requirements.txt",
    description="Returns the requirements.txt content as a downloadable file.",
    response_class=Response,
)
async def download_requirements(body: DownloadRequest):
    """Return the generated requirements.txt as a file download."""
    return Response(
        content=body.content,
        media_type="text/plain",
        headers={
            "Content-Disposition": 'attachment; filename="requirements.txt"',
        },
    )
