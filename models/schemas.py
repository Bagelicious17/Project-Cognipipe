"""
CogniPipe — Pydantic v2 Data Models
====================================

Defines every schema used across the three engine layers:

    DataProfiler  →  ProfileResult (JSON dict)  →  GeminiOrchestrator
    GeminiOrchestrator  →  GeminiResult (JSON dict)  →  CodeAssembler

This module is the single source of truth for all inter-layer contracts.
All models are fully JSON-serializable via ``model_dump()``.

Design decisions
----------------
1.  **Enums are ``str`` enums** so they serialize to human-readable strings
    instead of opaque integers.  This is critical because the JSON blob is
    forwarded as-is into Gemini prompts.

2.  **Every numeric stat defaults to ``None``** rather than ``0`` so the
    consumer can distinguish "not computed" from "legitimately zero."

3.  **``ColumnProfile`` is a flat union container** that holds *all* stat
    sub-models as optional fields.  We intentionally avoid inheritance or
    tagged-union discriminators here because every column shares the same
    metadata header (name, dtype, semantic type, missing info).  Whether
    ``numerical_stats`` or ``categorical_stats`` is populated depends on
    the inferred semantic type.  This keeps the profiler code simple and
    makes the JSON shape predictable for prompt engineering.

4.  **``ProfileResult`` is the only model the profiler returns.**  It
    contains both per-column profiles and dataset-level flags.  Later
    layers never need to reach back into the profiler.

5.  **Gemini-layer schemas** (``FeatureEngineeringPrescription``,
    ``MLArchitectureRecommendation``, ``GeminiResult``) are defined here
    too — even though we won't implement that layer yet — so that the
    JSON contract is locked in from day one.

6.  **CodeAssembler output schema** (``GeneratedPipeline``) is a thin
    wrapper around the generated code string + metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class SemanticType(str, Enum):
    """Inferred semantic role of a column."""

    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    CYCLICAL = "cyclical"
    DATETIME = "datetime"
    ID = "id"
    TARGET_CANDIDATE = "target_candidate"
    UNKNOWN = "unknown"


class MissingPattern(str, Enum):
    """Heuristic classification of missing-value mechanism.

    - ``NONE``  — no missing values at all.
    - ``MCAR`` — Missing Completely At Random (< 5 % missing,
      missingness uncorrelated with other columns).
    - ``MAR``  — Missing At Random (missingness correlates with at
      least one other numerical column, r > 0.3).
    - ``MNAR`` — Missing Not At Random (missing values concentrate
      in extreme quantiles of the column itself).
    """

    NONE = "none"
    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"


class TaskType(str, Enum):
    """Likely ML task type inferred from the dataset shape."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"


class CorrelationMethod(str, Enum):
    """Statistical method used to compute a correlation coefficient."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"


class LeakageReason(str, Enum):
    """Why a column is flagged as a potential data leakage risk."""

    PERFECT_CORRELATION_WITH_TARGET = "perfect_correlation_with_target"
    NEAR_PERFECT_CORRELATION_WITH_TARGET = "near_perfect_correlation_with_target"
    SUSPICIOUSLY_HIGH_CARDINALITY = "suspiciously_high_cardinality"
    NAME_SUGGESTS_POST_HOC = "name_suggests_post_hoc"
    DERIVED_FROM_TARGET = "derived_from_target"


# ──────────────────────────────────────────────────────────────────────
# Layer 1 — DataProfiler output models
# ──────────────────────────────────────────────────────────────────────

class MissingInfo(BaseModel):
    """Missing-value statistics for a single column."""

    missing_count: int = Field(
        ...,
        ge=0,
        description="Absolute number of missing (NaN / None) values.",
    )
    missing_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Missing values as a percentage of total rows.",
    )
    missing_pattern: MissingPattern = Field(
        ...,
        description="Heuristic mechanism classification (MCAR / MAR / MNAR / none).",
    )


class NumericalStats(BaseModel):
    """Descriptive statistics for a numerical column.

    Every field defaults to ``None`` so the profiler can skip stats
    that are undefined for degenerate columns (e.g. all-null, single
    unique value).
    """

    mean: float | None = Field(None, description="Arithmetic mean.")
    median: float | None = Field(None, description="50th percentile (median).")
    std: float | None = Field(None, description="Sample standard deviation.")
    min: float | None = Field(None, description="Minimum value.")
    max: float | None = Field(None, description="Maximum value.")
    skewness: float | None = Field(None, description="Fisher skewness coefficient.")
    kurtosis: float | None = Field(None, description="Fisher (excess) kurtosis.")
    q1: float | None = Field(None, description="25th percentile.")
    q3: float | None = Field(None, description="75th percentile.")
    iqr: float | None = Field(None, description="Interquartile range (Q3 − Q1).")
    outlier_count_iqr: int | None = Field(
        None,
        ge=0,
        description="Count of values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR].",
    )
    outlier_count_zscore: int | None = Field(
        None,
        ge=0,
        description="Count of values with |z-score| > 3.",
    )
    zero_count: int | None = Field(
        None,
        ge=0,
        description="Number of exactly-zero values.",
    )
    negative_count: int | None = Field(
        None,
        ge=0,
        description="Number of strictly negative values.",
    )
    is_likely_log_distributed: bool = Field(
        False,
        description=(
            "True when skewness > 1.5 AND all non-null values > 0, "
            "suggesting a log transform may normalise the distribution."
        ),
    )


class CategoricalStats(BaseModel):
    """Descriptive statistics for a categorical (nominal / ordinal) column."""

    cardinality: int = Field(
        ...,
        ge=0,
        description="Number of distinct non-null values.",
    )
    cardinality_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Cardinality divided by total row count.  "
            "Values close to 1.0 suggest the column is an ID."
        ),
    )
    top_10_values: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Up to 10 most frequent values mapped to their occurrence counts.  "
            "Keys are stringified category labels."
        ),
    )
    rare_category_count: int = Field(
        0,
        ge=0,
        description=(
            "Number of categories whose frequency is below 1 % of total rows."
        ),
    )


class DatetimeStats(BaseModel):
    """Descriptive statistics for a datetime column.

    Dates are stored as ISO-8601 strings so the model stays
    JSON-serializable without custom encoders.
    """

    min_date: str | None = Field(
        None,
        description="Earliest date/time value (ISO-8601 string).",
    )
    max_date: str | None = Field(
        None,
        description="Latest date/time value (ISO-8601 string).",
    )
    time_span_days: float | None = Field(
        None,
        ge=0.0,
        description="Number of days between min_date and max_date.",
    )
    inferred_frequency: str | None = Field(
        None,
        description=(
            "Inferred time-series frequency label "
            "(e.g. 'daily', 'weekly', 'monthly', 'irregular')."
        ),
    )


class ColumnProfile(BaseModel):
    """Complete profile for a single DataFrame column.

    This is a *flat union* container: exactly one of
    ``numerical_stats``, ``categorical_stats``, or
    ``datetime_stats`` will be populated based on the inferred
    ``semantic_type``.  The remaining two will be ``None``.
    """

    # ── Identity ──────────────────────────────────────────────────
    column_name: str = Field(
        ...,
        description="Original column header as it appears in the DataFrame.",
    )
    dtype: str = Field(
        ...,
        description="Pandas dtype as a string (e.g. 'float64', 'object').",
    )
    inferred_semantic_type: SemanticType = Field(
        ...,
        description="Heuristically inferred semantic role.",
    )

    # ── Missing values ────────────────────────────────────────────
    missing: MissingInfo = Field(
        ...,
        description="Missing-value statistics and pattern classification.",
    )

    # ── Type-specific stats (mutually exclusive in practice) ──────
    numerical_stats: NumericalStats | None = Field(
        None,
        description="Populated when the column is continuous, ordinal, or cyclical.",
    )
    categorical_stats: CategoricalStats | None = Field(
        None,
        description="Populated when the column is nominal, ordinal, or ID.",
    )
    datetime_stats: DatetimeStats | None = Field(
        None,
        description="Populated when the column is a datetime type.",
    )

    # ── Unique-value metadata (useful across all types) ───────────
    nunique: int = Field(
        0,
        ge=0,
        description="Number of distinct non-null values.",
    )
    sample_values: list[Any] = Field(
        default_factory=list,
        description=(
            "Up to 5 representative non-null values, useful for prompt "
            "context so Gemini can see what the data looks like."
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Dataset-level flag models
# ──────────────────────────────────────────────────────────────────────

class CorrelationPair(BaseModel):
    """A pair of columns with high correlation."""

    column_a: str = Field(..., description="First column name.")
    column_b: str = Field(..., description="Second column name.")
    pearson: float | None = Field(
        None,
        description="Pearson correlation coefficient (linear).",
    )
    spearman: float | None = Field(
        None,
        description="Spearman rank correlation coefficient (monotonic).",
    )
    method_flagged: CorrelationMethod = Field(
        ...,
        description="Which method triggered the flag (|r| > 0.7).",
    )


class DataLeakageRisk(BaseModel):
    """A column flagged as a potential source of data leakage."""

    column_name: str = Field(
        ...,
        description="Name of the suspicious column.",
    )
    reason: LeakageReason = Field(
        ...,
        description="Why this column was flagged.",
    )
    detail: str = Field(
        "",
        description="Human-readable explanation of the risk.",
    )


class DatasetFlags(BaseModel):
    """Dataset-level heuristic flags produced by the DataProfiler.

    These flags guide downstream layers (Gemini orchestrator,
    code assembler) in choosing appropriate strategies.
    """

    # ── Shape ─────────────────────────────────────────────────────
    num_rows: int = Field(..., ge=0, description="Total number of rows.")
    num_columns: int = Field(..., ge=0, description="Total number of columns.")

    # ── Time-series detection ─────────────────────────────────────
    is_time_series: bool = Field(
        False,
        description="True if the dataset appears to be time-ordered.",
    )
    has_datetime_index: bool = Field(
        False,
        description=(
            "True if there is a datetime column that could serve "
            "as the index (monotonic, no duplicates)."
        ),
    )

    # ── Special columns ──────────────────────────────────────────
    suspected_id_columns: list[str] = Field(
        default_factory=list,
        description="Column names that look like unique row identifiers.",
    )
    suspected_target_column: str | None = Field(
        None,
        description=(
            "Best-guess target column based on name heuristics and "
            "position (last column fallback)."
        ),
    )

    # ── Task type ─────────────────────────────────────────────────
    likely_task_type: TaskType = Field(
        TaskType.UNKNOWN,
        description="Heuristic guess at the ML task type.",
    )
    class_imbalance_ratio: float | None = Field(
        None,
        description=(
            "Ratio of majority class count to minority class count.  "
            "Only set when a classification task is suspected.  "
            "Values significantly > 1.0 indicate imbalance."
        ),
    )

    # ── Correlations & leakage ────────────────────────────────────
    high_correlation_pairs: list[CorrelationPair] = Field(
        default_factory=list,
        description="Pairs of columns with |correlation| > 0.7.",
    )
    data_leakage_risks: list[DataLeakageRisk] = Field(
        default_factory=list,
        description="Columns flagged as potential data leakage sources.",
    )

    # ── Duplicate detection ───────────────────────────────────────
    duplicate_row_count: int = Field(
        0,
        ge=0,
        description="Number of fully duplicated rows in the dataset.",
    )
    duplicate_row_percentage: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Duplicated rows as a percentage of total rows.",
    )

    # ── Memory footprint ─────────────────────────────────────────
    memory_usage_mb: float = Field(
        0.0,
        ge=0.0,
        description="Approximate in-memory size of the DataFrame in MB.",
    )


class ProfileResult(BaseModel):
    """Top-level output of the DataProfiler.

    This is the **single artefact** passed from Layer 1 (DataProfiler)
    to Layer 2 (GeminiOrchestrator) as a JSON dict.  It must be fully
    serializable via ``model_dump()`` and must never contain raw pandas
    objects.
    """

    columns: list[ColumnProfile] = Field(
        ...,
        description="Per-column profiling results, one entry per DataFrame column.",
    )
    dataset: DatasetFlags = Field(
        ...,
        description="Dataset-level heuristic flags and metadata.",
    )
    profiling_duration_seconds: float = Field(
        0.0,
        ge=0.0,
        description="Wall-clock time the profiler took to run (seconds).",
    )
    profiler_version: str = Field(
        "0.1.0",
        description="Semver string of the profiler that produced this result.",
    )


# ──────────────────────────────────────────────────────────────────────
# Layer 2 — GeminiOrchestrator output models
# ──────────────────────────────────────────────────────────────────────


class AnalystDiagnostic(BaseModel):
    """Chain 1 output: high-level dataset diagnostic from the AI analyst.

    This is an intermediate artefact consumed by Chains 2 and 3.
    It confirms / overrides heuristic guesses from the profiler and
    provides structured guidance for downstream chains.
    """

    diagnostic_summary: str = Field(
        ...,
        description=(
            "Plain-English summary of the dataset, referencing "
            "real column names and specific statistics."
        ),
    )
    confirmed_task_type: str = Field(
        ...,
        description=(
            "AI-confirmed ML task type. One of: binary_classification, "
            "multiclass_classification, regression, time_series, "
            "clustering, unknown."
        ),
    )
    confirmed_target_column: str | None = Field(
        None,
        description="AI-confirmed target column name, or null for unsupervised tasks.",
    )
    critical_issues: list[str] = Field(
        default_factory=list,
        description="List of specific data quality problems found.",
    )
    column_roles: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of column_name → role.  Roles: "
            "'feature', 'target', 'id', 'datetime_index', 'drop_candidate'."
        ),
    )
    priority_columns_for_engineering: list[str] = Field(
        default_factory=list,
        description="Columns that need the most feature engineering attention.",
    )


class FeatureStep(BaseModel):
    """A single feature engineering operation prescribed by Gemini."""

    step_order: int = Field(
        ...,
        ge=1,
        description="Execution order (1 = first).",
    )
    operation: str = Field(
        "",
        description=(
            "Short operation label, e.g. 'log_transform', "
            "'one_hot_encode', 'create_interaction'."
        ),
    )
    target_columns: list[str] = Field(
        default_factory=list,
        description="Column(s) this operation applies to.",
    )
    new_column_name: str | list[str] | None = Field(
        None,
        description="Name of the newly created feature(s), if applicable.",
    )

    @classmethod
    def _coerce_new_column_name(cls, v: Any) -> str | None:
        if isinstance(v, list):
            return ", ".join(str(x) for x in v) if v else None
        return v

    from pydantic import model_validator

    @model_validator(mode="before")
    @classmethod
    def _fix_nulls_and_lists(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if isinstance(data.get("new_column_name"), list):
                names = data["new_column_name"]
                data["new_column_name"] = ", ".join(str(x) for x in names) if names else None
            return {k: v for k, v in data.items() if v is not None}
        return data
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Operation-specific parameters, e.g. "
            "{'strategy': 'median'} for imputation, "
            "{'n_bins': 10} for binning."
        ),
    )
    sklearn_equivalent: str = Field(
        "",
        description=(
            "Fully qualified sklearn (or compatible) class, e.g. "
            "'sklearn.preprocessing.StandardScaler'. Empty string if the "
            "operation has no direct sklearn equivalent."
        ),
    )
    rationale: str = Field(
        "",
        description="Why this step is recommended (one sentence).",
    )
    code_snippet: str = Field(
        "",
        description="Python code snippet implementing this step.",
    )
    priority: str = Field(
        "medium",
        description="Importance: 'critical', 'high', 'medium', 'low'.",
    )


class FeatureEngineeringPrescription(BaseModel):
    """Full feature engineering plan returned by Gemini."""

    steps: list[FeatureStep] = Field(
        default_factory=list,
        description="Ordered list of feature engineering operations.",
    )
    summary: str = Field(
        "",
        description="High-level summary of the feature engineering strategy.",
    )
    estimated_feature_count_after: int | None = Field(
        None,
        description="Estimated total column count after all steps are applied.",
    )


class ModelCandidate(BaseModel):
    """A single ML model recommended by Gemini."""

    model_name: str = Field(
        "",
        description="Human-readable model name, e.g. 'XGBoost Classifier'.",
    )
    sklearn_class: str = Field(
        "",
        description=(
            "Fully qualified scikit-learn (or compatible) class path, "
            "e.g. 'xgboost.XGBClassifier'."
        ),
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Recommended starting hyperparameters.",
    )
    rationale: str = Field(
        "",
        description="Why this model is suitable for the dataset.",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Rank among candidates (1 = best).",
    )

    from pydantic import model_validator
    @model_validator(mode="before")
    @classmethod
    def _strip_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


class PreprocessingStep(BaseModel):
    """A preprocessing operation recommended before model training."""

    step_order: int = Field(..., ge=1, description="Execution order.")
    operation: str = Field(
        "",
        description="Operation label, e.g. 'impute_median', 'standard_scale'.",
    )
    target_columns: list[str] = Field(
        default_factory=list,
        description="Columns to apply this operation to.",
    )
    code_snippet: str = Field(
        "",
        description="Python code implementing the step.",
    )
    rationale: str = Field(
        "",
        description="Why this preprocessing step is needed.",
    )

    from pydantic import model_validator
    @model_validator(mode="before")
    @classmethod
    def _strip_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


class EvaluationStrategy(BaseModel):
    """Model evaluation plan recommended by Gemini."""

    validation_method: str = Field(
        "",
        description=(
            "Validation strategy, e.g. 'stratified_kfold', "
            "'time_series_split', 'holdout'."
        ),
    )
    n_splits: int | None = Field(
        None,
        ge=2,
        description="Number of cross-validation folds, if applicable.",
    )
    primary_metric: str = Field(
        "",
        description="Primary evaluation metric, e.g. 'f1_weighted', 'rmse'.",
    )
    secondary_metrics: list[str] = Field(
        default_factory=list,
        description="Additional metrics to report.",
    )
    rationale: str = Field(
        "",
        description="Why this evaluation strategy was chosen.",
    )

    from pydantic import model_validator
    @model_validator(mode="before")
    @classmethod
    def _strip_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


class MLArchitectureRecommendation(BaseModel):
    """Full ML architecture plan returned by Gemini."""

    task_type: TaskType = Field(
        ...,
        description="Confirmed ML task type for the dataset.",
    )
    target_column: str = Field(
        ...,
        description="Confirmed target column name.",
    )
    preprocessing: list[PreprocessingStep] = Field(
        default_factory=list,
        description="Ordered preprocessing pipeline steps.",
    )
    model_candidates: list[ModelCandidate] = Field(
        default_factory=list,
        description="Ranked list of recommended models.",
    )
    evaluation: EvaluationStrategy | None = Field(
        None,
        description="Recommended evaluation strategy.",
    )
    summary: str = Field(
        "",
        description="High-level summary of the ML architecture plan.",
    )


class ChainTokenUsage(BaseModel):
    """Token usage for a single Gemini chain call."""

    chain_name: str = Field(..., description="Chain identifier, e.g. 'chain_1_analyst'.")
    input_tokens: int = Field(0, ge=0, description="Prompt token count.")
    output_tokens: int = Field(0, ge=0, description="Completion token count.")
    total_tokens: int = Field(0, ge=0, description="Sum of input + output tokens.")


class GeminiResult(BaseModel):
    """Combined output of the GeminiOrchestrator (Layer 2).

    This is the artefact passed from Layer 2 to Layer 3
    (CodeAssembler).
    """

    analyst_diagnostic: AnalystDiagnostic = Field(
        ...,
        description="Chain 1 output: AI analyst diagnostic.",
    )
    feature_engineering: FeatureEngineeringPrescription = Field(
        ...,
        description="AI-generated feature engineering prescription.",
    )
    ml_architecture: MLArchitectureRecommendation = Field(
        ...,
        description="AI-generated ML architecture recommendation.",
    )
    raw_responses: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Raw JSON responses from each Gemini call, keyed by prompt "
            "name. Useful for debugging and auditing."
        ),
    )
    token_usage: list[ChainTokenUsage] = Field(
        default_factory=list,
        description="Per-chain token usage for cost tracking.",
    )
    orchestration_duration_seconds: float = Field(
        0.0,
        ge=0.0,
        description="Total wall-clock time for all Gemini calls (seconds).",
    )


# ──────────────────────────────────────────────────────────────────────
# Layer 3 — CodeAssembler output model
# ──────────────────────────────────────────────────────────────────────

class GeneratedPipeline(BaseModel):
    """Output of the CodeAssembler (Layer 3).

    Contains the generated Python pipeline code and metadata.
    The code is **never executed** on the server — it is returned
    to the user for download and local execution.
    """

    python_script: str = Field(
        ...,
        description="Complete, runnable .py pipeline script.",
    )
    notebook_json: str | None = Field(
        None,
        description=(
            "JSON string of an nbformat-compatible Jupyter notebook.  "
            "``None`` if notebook generation is not requested."
        ),
    )
    requirements_txt: str = Field(
        "",
        description="Generated pip requirements for the pipeline.",
    )
    pipeline_summary: str = Field(
        "",
        description="Human-readable summary of what the pipeline does.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of code generation.",
    )
    source_profile_version: str = Field(
        "",
        description="Profiler version that produced the input ProfileResult.",
    )
