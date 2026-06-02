<div align="center">

# 🧠 CogniPipe

**Automated Feature Engineering & ML Orchestration Platform**

*Drop a CSV. Get a production-ready ML pipeline.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gemini 2.5 Flash](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-orange.svg)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-00a393.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is CogniPipe?

CogniPipe takes a **raw CSV/Excel file** and automatically produces a complete, production-ready machine learning pipeline:

1. 📊 **Data Profiling Report** — Statistical analysis, semantic type inference, missing patterns, and leakage detection.
2. 🤖 **AI-Powered Feature Engineering** — Gemini-generated transformation prescriptions based on the dataset's unique shape and statistics.
3. 🏗️ **ML Architecture Recommendations** — Hyperparameter-tuned model selection, preprocessing workflows, and cross-validation strategies.
4. 📦 **Downloadable Pipeline** — A ready-to-run `.py` script, a fully formatted `.ipynb` Jupyter Notebook, and a custom `requirements.txt`.

> **Zero code required from the user. Zero data leaves your machine during execution (only statistical metadata is sent to Gemini).**

---

## Architecture

CogniPipe uses a **three-layer decoupled architecture** wrapped in a blazing-fast **FastAPI** web service. Layers communicate exclusively through strict JSON contracts (Pydantic v2):

```
┌─────────────┐      JSON       ┌─────────────────────┐      JSON       ┌────────────────┐
│ DataProfiler │ ──────────────► │ GeminiOrchestrator  │ ──────────────► │ CodeAssembler  │
│  (Layer 1)   │                 │     (Layer 2)        │                 │   (Layer 3)    │
│              │                 │                      │                 │                │
│ pandas       │                 │ Gemini API only      │                 │ String assembly│
│ scipy/numpy  │                 │ No pandas            │                 │ No AI, no pandas│
└─────────────┘                 └─────────────────────┘                 └────────────────┘
     ▲                                                                         │
     │                                                                         ▼
   Raw CSV                                                              Pipeline .py/.ipynb
```

For a deep technical dive into the orchestration mechanics, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/Project-Cognipipe.git
cd Project-Cognipipe
pip install -r requirements.txt
```

### 2. Set Up Your API Key

```bash
# Copy the template
cp .env.example .env

# Edit .env and paste your Gemini API key
# Get one at: https://aistudio.google.com/apikey
```

### 3. Start the FastAPI Server

```bash
python run.py
```
*The API will start at `http://localhost:8000`.*

---

## API Endpoints

Once the server is running, you can interact with the engine via REST endpoints.

### 🟢 `GET /health`
Check if the server is running and the Gemini API key is valid.

### 📊 `POST /api/v1/pipeline/profile-only`
Run **Layer 1 only**. Upload a dataset and get the `DataProfiler` output in JSON. Excellent for inspecting columns, leakage risks, and data flags before spending AI tokens.
```bash
curl -X POST http://localhost:8000/api/v1/pipeline/profile-only -F "file=@your_dataset.csv"
```

### 🚀 `POST /api/v1/pipeline/generate`
Run the **Full Pipeline**. Upload a dataset and receive the generated `python_script`, `notebook_json`, and `requirements_txt` as JSON strings.
```bash
curl -X POST http://localhost:8000/api/v1/pipeline/generate -F "file=@your_dataset.csv"
```

### 💾 `POST /api/v1/pipeline/download/*`
Download the generated artifacts directly as files (pass the generated content in the request body).
- `/download/script` ➡️ `pipeline.py`
- `/download/notebook` ➡️ `pipeline.ipynb`
- `/download/requirements` ➡️ `requirements.txt`

---

## Project Structure

```text
Project-Cognipipe/
├── .env.example           ← API key template
├── config.py              ← Centralized settings loader (API limits, timeouts)
├── run.py                 ← Uvicorn server entrypoint
├── app/
│   ├── main.py            ← FastAPI application factory & lifespan
│   └── routers/           
│       ├── pipeline.py    ← Core generation & download endpoints
│       └── schemas.py     ← API request/response models
├── models/
│   └── schemas.py         ← Pydantic v2 engine data models
├── services/
│   ├── data_profiler.py   ← Layer 1: Statistical profiling
│   ├── gemini_prompts.py  ← Prompt templates & chains
│   ├── gemini_orchestrator.py ← Layer 2: AI reasoning
│   └── code_assembler.py  ← Layer 3: Code & Notebook generation
└── tests/
    └── ...                ← 100 passing unit & integration tests
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI / Uvicorn | Async REST API orchestration |
| Data Models | Pydantic v2 | Type-safe schemas, automatic JSON serialization |
| Profiling | pandas, scipy, numpy | Statistical analysis, type inference |
| AI Engine | Gemini 1.5 Pro / 2.5 Flash | Feature engineering & ML architecture design |
| Code Gen | nbformat, string templates | Pipeline script & notebook assembly |

---

## Build Status

| Layer | Status | Tests |
|-------|--------|-------|
| Schemas (`models/schemas.py`) | ✅ Complete | Validated |
| Layer 1 — DataProfiler | ✅ Complete | 27/27 passing |
| Layer 2 — GeminiOrchestrator | ✅ Complete | 22/22 passing |
| Layer 3 — CodeAssembler | ✅ Complete | 15/15 passing |
| FastAPI Routes | ✅ Complete | 10/10 passing |
| Frontend | ⬜ Planned | — |

**Total Test Coverage:** 100/100 Passing ✅

---

## License

MIT — see [LICENSE](LICENSE) for details.
