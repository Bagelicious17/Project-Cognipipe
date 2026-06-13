"""
CogniPipe — API Request / Response Schemas
=============================================

Thin wrappers around the engine-layer Pydantic models.
These define the HTTP contract — what the client sends and receives.
Engine internals (ProfileResult, GeminiResult) stay internal.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = Field(
        ...,
        description="Server status, always 'healthy' if reachable.",
        examples=["healthy"],
    )
    version: str = Field(
        ...,
        description="CogniPipe API version.",
        examples=["0.1.0"],
    )
    gemini_model: str = Field(
        ...,
        description="Configured Gemini model name.",
        examples=["gemini-2.5-flash"],
    )


# ──────────────────────────────────────────────────────────────────────
# Pipeline generation
# ──────────────────────────────────────────────────────────────────────

class PipelineResponse(BaseModel):
    """POST /api/v1/pipeline/generate — successful response.

    Contains all generated artifacts as strings, plus timing metadata.
    The client can render these in a UI or use the download endpoints
    to retrieve them as files.
    """

    python_script: str = Field(
        ...,
        description="Complete, runnable .py pipeline script.",
    )
    notebook_json: str | None = Field(
        None,
        description="JSON string of a Jupyter notebook (.ipynb).",
    )
    requirements_txt: str = Field(
        ...,
        description="Generated pip requirements for the pipeline.",
    )
    pipeline_summary: str = Field(
        ...,
        description="Human-readable summary of the generated pipeline.",
    )
    generated_at: datetime = Field(
        ...,
        description="UTC timestamp of code generation.",
    )
    profiling_duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time for Layer 1 profiling (seconds).",
    )
    orchestration_duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time for Layer 2 Gemini calls (seconds).",
    )


# ──────────────────────────────────────────────────────────────────────
# Profile-only
# ──────────────────────────────────────────────────────────────────────

# ProfileResult from models.schemas is returned directly since it's
# already a Pydantic model. No wrapper needed — FastAPI serializes it
# natively.


# ──────────────────────────────────────────────────────────────────────
# Download requests
# ──────────────────────────────────────────────────────────────────────

class DownloadRequest(BaseModel):
    """Request body for download endpoints.

    The client passes back the generated content from the
    PipelineResponse so the server can return it as a downloadable file.
    This avoids server-side state / caching.
    """

    content: str = Field(
        ...,
        description="The file content to return as a download.",
    )


# ──────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Structured error response for all failure modes.

    The ``stage`` field tells the client *where* in the pipeline
    the failure occurred, enabling targeted retry logic.
    """

    error: str = Field(
        ...,
        description="Short error category.",
        examples=["validation_error", "orchestration_error", "timeout"],
    )
    detail: str = Field(
        ...,
        description="Human-readable error message with actionable guidance.",
        examples=["File must be .csv or .xlsx format."],
    )
    stage: str = Field(
        ...,
        description=(
            "Which pipeline stage failed. One of: "
            "'upload', 'profiling', 'orchestration', 'assembly'."
        ),
        examples=["upload"],
    )


# ──────────────────────────────────────────────────────────────────────
# Streaming events (NDJSON)
# ──────────────────────────────────────────────────────────────────────

class StreamProgressEvent(BaseModel):
    """Intermediate progress update streamed during pipeline generation.

    Sent as one JSON line in the NDJSON response from ``/generate``.
    """

    type: str = Field(
        "progress",
        description="Event discriminator, always 'progress'.",
    )
    progress: int = Field(
        ...,
        ge=0,
        le=100,
        description="Completion percentage (0–100).",
    )
    message: str = Field(
        ...,
        description="Human-readable status message for UI display.",
    )


class StreamDoneEvent(BaseModel):
    """Final event containing the complete pipeline output.

    Sent as one JSON line after 100% progress.
    """

    type: str = Field(
        "done",
        description="Event discriminator, always 'done'.",
    )
    data: PipelineResponse = Field(
        ...,
        description="The complete pipeline generation result.",
    )


class StreamErrorEvent(BaseModel):
    """Error event sent when any pipeline stage fails.

    Sent as one JSON line; the stream ends after this event.
    """

    type: str = Field(
        "error",
        description="Event discriminator, always 'error'.",
    )
    error: str = Field(
        ...,
        description="Short error category.",
    )
    detail: str = Field(
        ...,
        description="Human-readable error message.",
    )
    stage: str = Field(
        ...,
        description="Which pipeline stage failed.",
    )
