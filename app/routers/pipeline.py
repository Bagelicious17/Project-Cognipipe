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
import logging
from functools import partial

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response

from app.routers.schemas import (
    DownloadRequest,
    ErrorResponse,
    PipelineResponse,
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
                "detail": f"Data profiling failed: {e}",
                "stage": "profiling",
            },
        )

    return profile


# ──────────────────────────────────────────────────────────────────────
# POST /generate
# ──────────────────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=PipelineResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        504: {"model": ErrorResponse, "description": "Timeout"},
    },
    summary="Generate a full ML pipeline",
    description=(
        "Upload a CSV or XLSX file and receive a complete, runnable "
        "ML pipeline: Python script, Jupyter notebook, and "
        "requirements.txt. Runs all 3 engine layers."
    ),
)
async def generate_pipeline(file: UploadFile):
    """Run the full 3-layer pipeline: Profile → Gemini → Assemble."""
    df = await _validate_and_read_csv(file)
    loop = asyncio.get_event_loop()

    # Layer 1 — Profiling
    try:
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
                    "Try uploading a smaller dataset."
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
                "detail": f"Data profiling failed: {e}",
                "stage": "profiling",
            },
        )

    # Layer 2 — Gemini orchestration
    try:
        orchestrator = GeminiOrchestrator(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model_name,
        )
        gemini_result = await asyncio.wait_for(
            loop.run_in_executor(None, partial(orchestrator.run, profile)),
            timeout=settings.gemini_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "timeout",
                "detail": (
                    f"Gemini orchestration timed out after "
                    f"{settings.gemini_timeout_seconds}s. "
                    "The AI analysis is taking too long. "
                    "Try a smaller dataset or increase GEMINI_TIMEOUT_SECONDS."
                ),
                "stage": "orchestration",
            },
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "configuration_error",
                "detail": str(e),
                "stage": "orchestration",
            },
        )
    except Exception as e:
        logger.exception("Gemini orchestration failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "orchestration_error",
                "detail": f"AI analysis failed: {e}",
                "stage": "orchestration",
            },
        )

    # Layer 3 — Code assembly
    try:
        assembler = CodeAssembler()
        pipeline = assembler.build(profile, gemini_result)
    except Exception as e:
        logger.exception("Code assembly failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "assembly_error",
                "detail": f"Code assembly failed: {e}",
                "stage": "assembly",
            },
        )

    return PipelineResponse(
        python_script=pipeline.python_script,
        notebook_json=pipeline.notebook_json,
        requirements_txt=pipeline.requirements_txt,
        pipeline_summary=pipeline.pipeline_summary,
        generated_at=pipeline.generated_at,
        profiling_duration_seconds=profile.profiling_duration_seconds,
        orchestration_duration_seconds=gemini_result.orchestration_duration_seconds,
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
