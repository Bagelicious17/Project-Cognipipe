# 🏗️ CogniPipe Architecture

This document outlines the technical architecture of the CogniPipe backend engine.

## Core Principles

CogniPipe is built on three non-negotiable architectural principles:

1. **Decoupled Layers**: The engine is divided into three distinct layers. They communicate *exclusively* via strongly-typed JSON (Pydantic models).
2. **Zero Data Egress**: User data (DataFrames) *never* leaves the host machine. Only metadata (the JSON profile) is sent to the AI model.
3. **Output is Code**: The final artifact is a self-contained, downloadable Python script or Jupyter Notebook. The server does not execute the generated ML pipeline.

---

## The Three Layers

### Layer 1: DataProfiler (`services/data_profiler.py`)
**Input:** `pandas.DataFrame`
**Output:** `ProfileResult` (JSON)
**Rules:** ONLY pandas, numpy, scipy. No AI calls.

This layer performs deterministic statistical analysis and heuristic inference.
- **Semantic Type Inference**: Detects column roles (e.g., continuous, cyclical, ID, target candidate) using regex and cardinality heuristics.
- **Missing Value Analysis**: Classifies missing data mechanisms (MCAR, MAR, MNAR) using correlation and quantile analysis.
- **Statistical Summaries**: Computes descriptive statistics based on the inferred semantic type.
- **Dataset Flags**: Identifies task types, leakage risks, and highly correlated features.

### Layer 2: GeminiOrchestrator (`services/gemini_orchestrator.py`)
**Input:** `ProfileResult` (JSON)
**Output:** `GeminiResult` (JSON)
**Rules:** ONLY Gemini API. No pandas.

This layer acts as the AI reasoning engine. It takes the JSON metadata from Layer 1 and prompts Gemini 1.5 Pro to generate:
- **Feature Engineering Prescriptions**: A sequence of transformations (e.g., log transforms, interactions).
- **ML Architecture Recommendations**: Candidate models, preprocessing steps, and evaluation strategies tailored to the dataset.
The output is strictly enforced as structured JSON.

### Layer 3: CodeAssembler (`services/code_assembler.py`)
**Input:** `GeminiResult` (JSON) + `ProfileResult` (JSON)
**Output:** `GeneratedPipeline` (String/JSON)
**Rules:** ONLY string assembly/templating. No AI, no pandas.

This layer maps the AI prescriptions to executable Python code.
- Uses the `FeatureStep` definitions (which include `target_columns` and `parameters`) to generate specific `pandas` and `scikit-learn` code snippets.
- Combines the steps into a cohesive, runnable script or `nbformat` compatible Jupyter Notebook.

---

## Data Models (`models/schemas.py`)

The schemas act as the single source of truth for inter-layer communication.

### Key Decisions:
- **Flat Unions**: `ColumnProfile` contains optional fields for numerical, categorical, and datetime stats, avoiding complex polymorphic dispatch.
- **Enums as Strings**: Enums serialize to strings to ensure the JSON sent to Gemini remains human-readable.
- **None Defaults**: Numeric stats default to `None` to distinguish between "not applicable/calculated" and a legitimate zero.

## Security & Privacy

By relying on Layer 1 to extract structural metadata and statistical aggregates, CogniPipe ensures that actual row-level PII or sensitive data is never transmitted to the Gemini API. The orchestrator reasons entirely over metadata.
