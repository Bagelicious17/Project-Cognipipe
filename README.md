<div align="center">

# 🧠 CogniPipe

**Automated Feature Engineering & ML Orchestration Platform**

*Drop a CSV. Get a production-ready ML pipeline.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gemini 1.5 Pro](https://img.shields.io/badge/AI-Gemini%201.5%20Pro-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is CogniPipe?

CogniPipe takes a **raw CSV/Excel file** and produces:

1. 📊 **Data Profiling Report** — statistical analysis, type inference, missing patterns, leakage detection
2. 🤖 **AI-Powered Feature Engineering** — Gemini-generated transformation prescriptions
3. 🏗️ **ML Architecture Recommendations** — model selection, preprocessing, evaluation strategy
4. 📦 **Downloadable Pipeline Script** — production-ready `.py` + `.ipynb` you run locally

> **Zero code required from the user. Zero data leaves your machine during execution.**

---

## Architecture

CogniPipe uses a **three-layer decoupled architecture** where each layer communicates only through JSON:

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

For a deep technical dive, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

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

### 3. Try the Profiler (Layer 1 — works now!)

```python
import pandas as pd
from services.data_profiler import DataProfiler

df = pd.read_csv("your_dataset.csv")
profiler = DataProfiler()
result = profiler.profile(df)

# Full JSON report
print(result.model_dump_json(indent=2))

# Quick summary
print(f"Task type:  {result.dataset.likely_task_type.value}")
print(f"Target:     {result.dataset.suspected_target_column}")
print(f"Leakage:    {[r.column_name for r in result.dataset.data_leakage_risks]}")
```

---

## Project Structure

```
Project-Cognipipe/
├── .env.example           ← API key template (copy to .env)
├── config.py              ← Centralized settings loader
├── ARCHITECTURE.md        ← Technical deep-dive
├── models/
│   ├── __init__.py
│   └── schemas.py         ← All Pydantic v2 data models
├── services/
│   ├── __init__.py
│   ├── data_profiler.py   ← Layer 1: Statistical profiling
│   ├── gemini_prompts.py  ← Prompt templates (coming)
│   ├── gemini_orchestrator.py  ← Layer 2: AI reasoning (coming)
│   └── code_assembler.py  ← Layer 3: Code generation (coming)
└── tests/
    ├── __init__.py
    └── test_data_profiler.py  ← 39 tests, all passing
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Models | Pydantic v2 | Type-safe schemas, JSON serialization |
| Profiling | pandas, scipy, numpy | Statistical analysis, no AI |
| AI Engine | Gemini 1.5 Pro | Feature engineering & ML recommendations |
| Code Gen | nbformat, string templates | Pipeline script assembly |
| API (future) | FastAPI | REST endpoints |
| Config | python-dotenv | Environment variable management |

---

## Build Status

| Layer | Status | Tests |
|-------|--------|-------|
| Schemas (`models/schemas.py`) | ✅ Complete | Validated |
| Layer 1 — DataProfiler | ✅ Complete | 39/39 passing |
| Layer 2 — GeminiOrchestrator | 🔜 Next | — |
| Layer 3 — CodeAssembler | ⬜ Planned | — |
| FastAPI Routes | ⬜ Planned | — |
| Frontend | ⬜ Planned | — |

---

## Requirements

```
python>=3.11
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
pydantic>=2.0
python-dotenv>=1.0
google-generativeai>=0.5
nbformat>=5.9
fastapi>=0.110
uvicorn>=0.29
pytest>=8.0
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
