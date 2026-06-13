"""
Tests for the FastAPI integration.
"""


from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from models.schemas import (
    ProfileResult,
    DatasetFlags,
    TaskType,
)

app = create_app()
client = TestClient(app)

# 11 rows, 3 columns to pass the >= 10 rows and >= 2 columns check
TINY_CSV = b"""ColA,ColB,ColC
1,A,10.0
2,B,20.0
3,A,30.0
4,B,40.0
5,A,50.0
6,B,60.0
7,A,70.0
8,B,80.0
9,A,90.0
10,B,100.0
"""


def test_health_endpoint():
    """Returns 200 with status 'healthy'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "gemini_model" in data


def test_upload_non_csv_returns_400():
    """Rejects non-CSV files."""
    response = client.post(
        "/api/v1/pipeline/profile-only",
        files={"file": ("test.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["stage"] == "upload"
    assert "Unsupported file extension" in response.json()["detail"]["detail"]


def test_upload_empty_csv_returns_400():
    """Rejects empty/headerless CSV."""
    response = client.post(
        "/api/v1/pipeline/profile-only",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]["detail"].lower()


@patch("app.routers.pipeline._MAX_BYTES", 0)
def test_upload_too_large_returns_413():
    """Rejects files over size limit. We mock _MAX_BYTES to 0."""
    response = client.post(
        "/api/v1/pipeline/profile-only",
        files={"file": ("large.csv", TINY_CSV, "text/csv")},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["error"] == "file_too_large"


def test_upload_too_small_returns_400():
    """Rejects files with less than 10 rows."""
    tiny = b"A,B\n1,2\n3,4\n"
    response = client.post(
        "/api/v1/pipeline/profile-only",
        files={"file": ("small.csv", tiny, "text/csv")},
    )
    assert response.status_code == 400
    assert "minimum of 10 rows" in response.json()["detail"]["detail"]


def test_generate_without_file_returns_422():
    """Missing file param."""
    response = client.post("/api/v1/pipeline/generate")
    assert response.status_code == 422


@patch("services.data_profiler.DataProfiler.profile")
def test_profile_only_endpoint(mock_profile):
    """Profile-only returns valid ProfileResult JSON."""
    mock_profile.return_value = ProfileResult(
        columns=[],
        dataset=DatasetFlags(
            num_rows=10,
            num_columns=3,
            likely_task_type=TaskType.BINARY_CLASSIFICATION,
        ),
    )
    response = client.post(
        "/api/v1/pipeline/profile-only",
        files={"file": ("data.csv", TINY_CSV, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["dataset"]["num_rows"] == 10


@pytest.mark.slow
def test_generate_pipeline_with_csv():
    """Full end-to-end with a tiny CSV fixture. Makes real Gemini calls.

    The /generate endpoint now returns NDJSON streaming, so we read
    the response as lines, parse each JSON event, and verify:
    - At least one 'progress' event was received
    - Progress percentages are non-decreasing
    - The final 'done' event contains the pipeline payload
    """
    import json

    with client.stream(
        "POST",
        "/api/v1/pipeline/generate",
        files={"file": ("data.csv", TINY_CSV, "text/csv")},
    ) as response:
        assert response.status_code == 200

        events = []
        for line in response.iter_lines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    # Must have at least 1 progress + 1 done
    progress_events = [e for e in events if e["type"] == "progress"]
    done_events = [e for e in events if e["type"] == "done"]

    assert len(progress_events) >= 1, "Expected at least one progress event"
    assert len(done_events) == 1, "Expected exactly one done event"

    # Progress percentages should be non-decreasing
    pcts = [e["progress"] for e in progress_events]
    assert pcts == sorted(pcts), f"Progress not monotonic: {pcts}"

    # Done event contains the pipeline
    data = done_events[0]["data"]
    assert "python_script" in data
    assert "requirements_txt" in data
    assert "pipeline_summary" in data


def test_download_script():
    """Test the script download endpoint."""
    response = client.post(
        "/api/v1/pipeline/download/script",
        json={"content": "print('hello')"}
    )
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="pipeline.py"'
    assert response.text == "print('hello')"


def test_download_notebook():
    """Test the notebook download endpoint."""
    response = client.post(
        "/api/v1/pipeline/download/notebook",
        json={"content": "{}"}
    )
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="pipeline.ipynb"'


def test_download_requirements():
    """Test the requirements download endpoint."""
    response = client.post(
        "/api/v1/pipeline/download/requirements",
        json={"content": "pandas"}
    )
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="requirements.txt"'
