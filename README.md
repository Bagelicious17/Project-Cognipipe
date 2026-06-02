<div align="center">

# 📓 CogniPipe

**AI-Powered ML Pipeline Generator**

*Drop a CSV. Get a production-ready machine learning pipeline — no code required.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gemini AI](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-orange.svg)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Tests](https://img.shields.io/badge/tests-103%20passing-brightgreen.svg)](#test-coverage)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Live Demo →](https://cognipipe-frontend.vercel.app)**

</div>

---

## What is CogniPipe?

CogniPipe is an end-to-end platform that transforms **raw CSV/Excel files** into complete, runnable machine learning pipelines — entirely automated by AI. It's designed for data scientists, students, and ML engineers who want to skip repetitive boilerplate and focus on what matters.

### What You Get

| Step | Output |
|------|--------|
| 📊 **Data Profiling** | Statistical analysis, semantic type inference, missing-value pattern detection (MCAR/MAR/MNAR), data leakage warnings, and correlation analysis |
| 🤖 **AI Feature Engineering** | Gemini-generated transformation prescriptions tailored to your dataset's unique shape and statistics |
| 🏗️ **ML Architecture Design** | Model selection with hyperparameter recommendations, preprocessing workflows, and cross-validation strategies |
| 📦 **Downloadable Pipeline** | A ready-to-run `.py` script, a fully formatted `.ipynb` Jupyter Notebook, and a custom `requirements.txt` |

> **🔒 Privacy First:** Your raw data never leaves your machine. Only statistical metadata (column types, distributions, correlations) is sent to the AI model — never actual row-level data.

---

## Architecture

CogniPipe uses a **three-layer decoupled architecture** where each layer communicates exclusively through strict JSON contracts (Pydantic v2 models):

```
                          ┌──────────────────────────────────────────────────────────┐
                          │                    FastAPI Server                        │
                          │                                                         │
  ┌──────────┐     CSV    │  ┌──────────────┐   JSON   ┌───────────────────┐  JSON  │   ┌──────────────┐
  │  React   │ ─────────► │  │ DataProfiler │ ───────► │ GeminiOrchestrator│ ─────► │   │ CodeAssembler│
  │ Frontend │ ◄───────── │  │  (Layer 1)   │          │    (Layer 2)      │        │   │  (Layer 3)   │
  └──────────┘    JSON    │  │              │          │                   │        │   │              │
   Vercel                 │  │ pandas       │          │ Gemini API only   │        │   │ String-based │
                          │  │ scipy/numpy  │          │ No pandas         │        │   │ No AI        │
                          │  └──────────────┘          └───────────────────┘        │   └──────┬───────┘
                          │       Railway                                           │          │
                          └──────────────────────────────────────────────────────────┘          ▼
                                                                                       .py / .ipynb
                                                                                     requirements.txt
```

| Layer | Responsibility | Dependencies | I/O |
|-------|---------------|-------------|-----|
| **Layer 1 — DataProfiler** | Statistical profiling, semantic type inference, missing-value analysis | pandas, scipy, numpy | `DataFrame` → `ProfileResult` (JSON) |
| **Layer 2 — GeminiOrchestrator** | AI reasoning via 3-chain prompting (Analyst → Feature Engineer → Architect) | Gemini API only | `ProfileResult` → `GeminiResult` (JSON) |
| **Layer 3 — CodeAssembler** | Deterministic code generation from AI prescriptions | String templates, nbformat | `GeminiResult` + `ProfileResult` → `.py` / `.ipynb` |

For a deep technical dive, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Live Demo

| Service | URL | Platform |
|---------|-----|----------|
| 🌐 Frontend | [cognipipe-frontend.vercel.app](https://cognipipe-frontend.vercel.app) | Vercel |
| ⚡ Backend API | [cognipipe-production.up.railway.app](https://cognipipe-production.up.railway.app) | Railway |

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- A [Gemini API key](https://aistudio.google.com/apikey)

### 1. Clone & Install Backend

```bash
git clone https://github.com/Bagelicious17/Project-Cognipipe.git
cd Project-Cognipipe
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your Gemini API key:
```env
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL_NAME=gemini-2.0-flash
```

### 3. Start the Backend

```bash
python run.py
```

The API starts at `http://localhost:8080`. Interactive docs are available at `http://localhost:8080/docs`.

### 4. Start the Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

The frontend starts at `http://localhost:5173` and connects to the local backend automatically.

---

## API Reference

### `GET /` — Health Check
Returns server status, version, and configured Gemini model.

### `POST /api/v1/pipeline/profile-only` — Data Profiling
Run **Layer 1 only**. Upload a CSV/XLSX and get the full `ProfileResult` JSON. Useful for inspecting column types, leakage risks, and data quality before committing AI tokens.
```bash
curl -X POST http://localhost:8080/api/v1/pipeline/profile-only \
  -F "file=@your_dataset.csv"
```

### `POST /api/v1/pipeline/generate` — Full Pipeline Generation
Run all 3 layers. Upload a dataset and receive the complete generated pipeline.
```bash
curl -X POST http://localhost:8080/api/v1/pipeline/generate \
  -F "file=@your_dataset.csv"
```

### `POST /api/v1/pipeline/download/*` — Download Artifacts
Download generated artifacts as files:
| Endpoint | Output |
|----------|--------|
| `/download/script` | `pipeline.py` |
| `/download/notebook` | `pipeline.ipynb` |
| `/download/requirements` | `requirements.txt` |

---

## Project Structure

```text
Project-Cognipipe/
├── .env.example               # Environment variable template
├── config.py                  # Centralized settings (API limits, timeouts, CORS)
├── run.py                     # Uvicorn server entrypoint
├── Dockerfile                 # Multi-stage production container
├── cloudbuild.yaml            # Google Cloud Build CI/CD
│
├── app/                       # FastAPI application
│   ├── main.py                # App factory, lifespan, health check
│   └── routers/
│       ├── pipeline.py        # Core pipeline endpoints + error sanitization
│       └── schemas.py         # API request/response models
│
├── models/
│   └── schemas.py             # Pydantic v2 engine data models (50+ models)
│
├── services/                  # Three-layer engine
│   ├── data_profiler.py       # Layer 1: Statistical profiling
│   ├── gemini_prompts.py      # Prompt templates & chain definitions
│   ├── gemini_orchestrator.py # Layer 2: AI reasoning (3-chain orchestration)
│   └── code_assembler.py      # Layer 3: Code & Notebook generation
│
├── frontend/                  # React SPA (Vite + Tailwind CSS v4)
│   ├── src/
│   │   ├── App.jsx            # Stage-based UI controller + dark mode
│   │   ├── api.js             # Backend API client
│   │   └── components/        # Upload, Profiling, Generating, Results, Error
│   └── DESIGN.md              # CodeNotebook design system spec
│
└── tests/                     # 103 passing tests
    ├── test_data_profiler.py   # 42 tests — profiling & type inference
    ├── test_code_assembler.py  # 29 tests — code generation
    ├── test_gemini_orchestrator.py # 22 tests — AI orchestration
    └── test_api.py             # 10 tests — FastAPI endpoints
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19, Vite 8, Tailwind CSS 4 | CodeNotebook-themed responsive UI with light/dark mode |
| **Backend** | FastAPI, Uvicorn | Async REST API with structured error handling |
| **Data Models** | Pydantic v2 | 50+ type-safe schemas with model validators |
| **Profiling** | pandas, scipy, numpy | Statistical analysis, semantic type inference, missing-value patterns |
| **AI Engine** | Google Gemini 2.0 Flash | 3-chain prompt orchestration for feature engineering & ML design |
| **Code Gen** | nbformat, string templates | Pipeline script & Jupyter Notebook assembly |
| **Deployment** | Railway (backend), Vercel (frontend) | Auto-deploy from GitHub on push |
| **Containerization** | Docker (multi-stage) | Production-ready image with non-root user |

---

## Test Coverage

All tests run in < 25 seconds with zero external API calls (Gemini is fully mocked).

```
$ pytest tests/ -m "not slow" -q

103 passed, 1 deselected, 2 warnings in 22.97s
```

| Test Suite | Count | Covers |
|-----------|-------|--------|
| `test_data_profiler.py` | 42 | Semantic type inference, missing patterns, numerical/categorical/datetime stats, dataset flags, edge cases |
| `test_code_assembler.py` | 29 | Script & notebook generation, sklearn mapping, requirements.txt output |
| `test_gemini_orchestrator.py` | 22 | 3-chain orchestration, retry logic, JSON parsing, token tracking |
| `test_api.py` | 10 | FastAPI endpoint validation, file upload, error responses |
| **Total** | **103** | |

---

## Key Features

### 🧠 Intelligent Data Profiling
- **Semantic Type Inference**: Automatically classifies columns as continuous, ordinal, cyclical, datetime, ID, or target candidate using regex + cardinality heuristics
- **Missing Value Diagnosis**: Determines if missing data is MCAR, MAR, or MNAR using correlation and quantile analysis
- **Data Leakage Detection**: Flags columns with suspiciously high correlation to the target or post-hoc naming patterns
- **Task Type Detection**: Infers binary classification, multiclass, regression, time series, or clustering from data shape

### 🤖 3-Chain AI Orchestration
1. **Chain 1 — Analyst**: Reviews the profile and produces a diagnostic summary with confirmed task type and column roles
2. **Chain 2 — Feature Engineer**: Prescribes ordered transformation steps (log transforms, interactions, encodings) with specific sklearn equivalents
3. **Chain 3 — ML Architect**: Recommends ranked model candidates with hyperparameters, preprocessing pipelines, and evaluation strategies

### 🎨 CodeNotebook UI
- Clean, warm-toned design inspired by physical notebooks
- Manual light/dark mode toggle with `localStorage` persistence
- Stage-based wizard flow: Upload → Profile Review → Generate → Download
- Animated striped progress bars during AI processing
- User-friendly error messages (raw API errors are never exposed)

### 🔐 Security & Privacy
- Raw row-level data never leaves the host — only statistical metadata is sent to Gemini
- Non-root Docker container for production deployments
- API key validation at startup
- Sanitized error responses — no internal details leak to the frontend

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key |
| `GEMINI_MODEL_NAME` | ❌ | `gemini-2.0-flash` | Gemini model to use |
| `GEMINI_TIMEOUT_SECONDS` | ❌ | `300` | Timeout for AI calls |
| `MAX_UPLOAD_MB` | ❌ | `50` | Max upload file size (MB) |
| `CORS_ORIGINS` | ❌ | `*` | Allowed CORS origins (comma-separated) |
| `API_HOST` | ❌ | `0.0.0.0` | Server bind address |
| `API_PORT` | ❌ | `8080` | Server port |

---

## Deployment

### Docker
```bash
docker build -t cognipipe .
docker run -p 8080:8080 -e GEMINI_API_KEY=your-key cognipipe
```

### Railway (Backend)
The backend auto-deploys from the `feat/fastapi-and-frontend` branch. Set `GEMINI_API_KEY` in Railway's Variables tab.

### Vercel (Frontend)
The frontend auto-deploys from the same branch with root directory set to `frontend/`. Set `VITE_API_URL` to point to the Railway backend URL.

---

## License

MIT — see [LICENSE](LICENSE) for details.
