"""
Unit tests for ``services.data_profiler.DataProfiler``.

Tests cover:
- Per-column profiling (numeric, categorical, datetime, edge cases)
- Semantic type inference heuristics
- Missing pattern classification (none / MCAR / MAR / MNAR)
- Dataset-level flag computation
- JSON serialization round-trip
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.schemas import (
    MissingPattern,
    SemanticType,
    TaskType,
)
from services.data_profiler import DataProfiler


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def profiler() -> DataProfiler:
    """Fresh profiler instance."""
    return DataProfiler()


@pytest.fixture
def basic_df() -> pd.DataFrame:
    """Realistic multi-type DataFrame for integration tests."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.randint(18, 80, n),
        "income": np.random.lognormal(10, 1, n).round(2),
        "gender": np.random.choice(["M", "F", "Other"], n, p=[0.45, 0.45, 0.1]),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "purchase_count": np.random.poisson(5, n),
        "satisfaction": np.random.choice([1, 2, 3, 4, 5], n),
        "churn": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })
    # Inject missing values
    df.loc[df.sample(10, random_state=1).index, "income"] = np.nan
    df.loc[df.sample(3, random_state=2).index, "gender"] = None
    return df


# ── ProfileResult shape ──────────────────────────────────────────────

class TestProfileResultShape:
    """Verify the overall ProfileResult structure."""

    def test_all_columns_profiled(self, profiler: DataProfiler, basic_df: pd.DataFrame):
        """Every DataFrame column must appear in the result."""
        result = profiler.profile(basic_df)
        assert len(result.columns) == len(basic_df.columns)

    def test_column_names_match(self, profiler: DataProfiler, basic_df: pd.DataFrame):
        result = profiler.profile(basic_df)
        names = [cp.column_name for cp in result.columns]
        assert names == list(basic_df.columns)

    def test_dataset_shape(self, profiler: DataProfiler, basic_df: pd.DataFrame):
        result = profiler.profile(basic_df)
        assert result.dataset.num_rows == len(basic_df)
        assert result.dataset.num_columns == len(basic_df.columns)

    def test_json_round_trip(self, profiler: DataProfiler, basic_df: pd.DataFrame):
        """model_dump() must produce a JSON-serializable dict."""
        result = profiler.profile(basic_df)
        d = result.model_dump()
        assert isinstance(d, dict)
        import json
        json.dumps(d, default=str)  # must not raise


# ── Semantic type inference ───────────────────────────────────────────

class TestSemanticTypeInference:
    """Verify heuristic rules for column classification."""

    def test_id_by_name(self, profiler: DataProfiler):
        df = pd.DataFrame({"user_id": [1, 2, 3], "value": [10, 20, 30]})
        result = profiler.profile(df)
        cp = result.columns[0]
        assert cp.inferred_semantic_type == SemanticType.ID

    def test_id_by_cardinality(self, profiler: DataProfiler):
        """High cardinality ratio (>0.95) → ID."""
        df = pd.DataFrame({"code": [f"x{i}" for i in range(100)]})
        result = profiler.profile(df)
        assert result.columns[0].inferred_semantic_type == SemanticType.ID

    def test_datetime_by_dtype(self, profiler: DataProfiler):
        df = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=10)})
        result = profiler.profile(df)
        assert result.columns[0].inferred_semantic_type == SemanticType.DATETIME

    def test_datetime_by_name(self, profiler: DataProfiler):
        """Name containing 'date' should trigger datetime, even if dtype is object."""
        df = pd.DataFrame({"order_date": ["2020-01-01", "2020-01-02", "2020-01-03"]})
        result = profiler.profile(df)
        assert result.columns[0].inferred_semantic_type == SemanticType.DATETIME

    def test_datetime_beats_id_cardinality(self, profiler: DataProfiler):
        """A datetime column with unique values should NOT be classified as ID."""
        df = pd.DataFrame({
            "signup_date": pd.date_range("2020-01-01", periods=50, freq="D"),
            "val": range(50),
        })
        result = profiler.profile(df)
        dt_col = [c for c in result.columns if c.column_name == "signup_date"][0]
        assert dt_col.inferred_semantic_type == SemanticType.DATETIME

    def test_target_candidate_by_name(self, profiler: DataProfiler):
        df = pd.DataFrame({"churn": [0, 1, 0], "x": [1, 2, 3]})
        result = profiler.profile(df)
        cp = [c for c in result.columns if c.column_name == "churn"][0]
        assert cp.inferred_semantic_type == SemanticType.TARGET_CANDIDATE

    def test_ordinal_low_cardinality_int(self, profiler: DataProfiler):
        df = pd.DataFrame({"rating": [1, 2, 3, 4, 5] * 20})
        result = profiler.profile(df)
        assert result.columns[0].inferred_semantic_type == SemanticType.ORDINAL

    def test_nominal_object_dtype(self, profiler: DataProfiler):
        df = pd.DataFrame({"color": ["red", "blue", "green"] * 40})
        result = profiler.profile(df)
        assert result.columns[0].inferred_semantic_type == SemanticType.NOMINAL

    def test_continuous_float(self, profiler: DataProfiler):
        # Use rounded floats so cardinality < n_rows (avoids ID by cardinality)
        df = pd.DataFrame({"weight": np.round(np.random.random(200), 1)})
        result = profiler.profile(df)
        assert result.columns[0].inferred_semantic_type == SemanticType.CONTINUOUS


# ── Numerical stats ───────────────────────────────────────────────────

class TestNumericalStats:
    """Verify numerical stat computations."""

    def test_basic_stats(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = profiler.profile(df)
        ns = result.columns[0].numerical_stats
        assert ns is not None
        assert ns.mean == pytest.approx(3.0, abs=0.01)
        assert ns.median == pytest.approx(3.0, abs=0.01)
        assert ns.min == pytest.approx(1.0)
        assert ns.max == pytest.approx(5.0)

    def test_outlier_detection(self, profiler: DataProfiler):
        values = list(range(100)) + [1000]  # 1000 is an outlier
        df = pd.DataFrame({"x": [float(v) for v in values]})
        result = profiler.profile(df)
        ns = result.columns[0].numerical_stats
        assert ns is not None
        assert ns.outlier_count_iqr >= 1
        assert ns.outlier_count_zscore >= 1

    def test_log_distribution_flag(self, profiler: DataProfiler):
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.lognormal(5, 2, 500)})
        result = profiler.profile(df)
        ns = result.columns[0].numerical_stats
        assert ns is not None
        assert ns.is_likely_log_distributed is True

    def test_all_null_column(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": [np.nan] * 10})
        result = profiler.profile(df)
        cp = result.columns[0]
        assert cp.missing.missing_count == 10
        assert cp.missing.missing_percentage == 100.0

    def test_single_value_column(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": [42.0] * 50})
        result = profiler.profile(df)
        ns = result.columns[0].numerical_stats
        assert ns is not None
        assert ns.std == pytest.approx(0.0)
        assert ns.outlier_count_zscore == 0

    def test_zero_and_negative_counts(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0]})
        result = profiler.profile(df)
        ns = result.columns[0].numerical_stats
        assert ns is not None
        assert ns.zero_count == 2
        assert ns.negative_count == 2


# ── Categorical stats ────────────────────────────────────────────────

class TestCategoricalStats:
    """Verify categorical stat computations."""

    def test_cardinality(self, profiler: DataProfiler):
        df = pd.DataFrame({"color": ["red", "blue", "green"] * 40})
        result = profiler.profile(df)
        cs = result.columns[0].categorical_stats
        assert cs is not None
        assert cs.cardinality == 3

    def test_top_10_values(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": ["a"] * 50 + ["b"] * 30 + ["c"] * 20})
        result = profiler.profile(df)
        cs = result.columns[0].categorical_stats
        assert cs is not None
        assert "a" in cs.top_10_values
        assert cs.top_10_values["a"] == 50

    def test_rare_categories(self, profiler: DataProfiler):
        # 1000 rows, categories with < 1% (< 10) occurrences are rare
        vals = ["common"] * 990 + ["rare1"] * 5 + ["rare2"] * 5
        df = pd.DataFrame({"x": vals})
        result = profiler.profile(df)
        cs = result.columns[0].categorical_stats
        assert cs is not None
        assert cs.rare_category_count == 2


# ── Datetime stats ────────────────────────────────────────────────────

class TestDatetimeStats:
    """Verify datetime stat computations."""

    def test_basic_datetime(self, profiler: DataProfiler):
        df = pd.DataFrame({
            "dt": pd.date_range("2023-01-01", periods=30, freq="D"),
        })
        result = profiler.profile(df)
        ds = result.columns[0].datetime_stats
        assert ds is not None
        assert ds.time_span_days == pytest.approx(29.0, abs=0.1)
        assert ds.inferred_frequency == "daily"


# ── Missing pattern classification ───────────────────────────────────

class TestMissingPatterns:
    """Verify MCAR / MAR / MNAR heuristics."""

    def test_no_missing(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = profiler.profile(df)
        assert result.columns[0].missing.missing_pattern == MissingPattern.NONE

    def test_mcar_low_missing(self, profiler: DataProfiler):
        """< 5% random missing → MCAR."""
        np.random.seed(42)
        x = np.random.random(200)
        x[np.random.choice(200, 5, replace=False)] = np.nan  # 2.5%
        df = pd.DataFrame({"x": x, "y": np.random.random(200)})
        result = profiler.profile(df)
        cp = [c for c in result.columns if c.column_name == "x"][0]
        assert cp.missing.missing_pattern == MissingPattern.MCAR


# ── Dataset-level flags ──────────────────────────────────────────────

class TestDatasetFlags:
    """Verify dataset-level heuristic flags."""

    def test_binary_classification(self, profiler: DataProfiler):
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.random(100),
            "target": np.random.choice([0, 1], 100),
        })
        result = profiler.profile(df)
        assert result.dataset.likely_task_type == TaskType.BINARY_CLASSIFICATION
        assert result.dataset.class_imbalance_ratio is not None

    def test_regression(self, profiler: DataProfiler):
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.random(100),
            "price": np.random.random(100) * 1000,
        })
        result = profiler.profile(df)
        assert result.dataset.suspected_target_column == "price"
        assert result.dataset.likely_task_type == TaskType.REGRESSION

    def test_target_tiebreaker_picks_lowest_cardinality(self, profiler: DataProfiler):
        """When multiple columns match target-name heuristics, prefer lowest cardinality."""
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.random(200),
            "revenue": np.random.random(200) * 10000,   # high cardinality
            "churn": np.random.choice([0, 1], 200),      # low cardinality
        })
        result = profiler.profile(df)
        # Both 'revenue' and 'churn' match TARGET_PAT, but churn has nunique=2
        assert result.dataset.suspected_target_column == "churn"

    def test_time_series_detection(self, profiler: DataProfiler):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "value": np.random.random(100),
        })
        result = profiler.profile(df)
        assert result.dataset.is_time_series is True
        assert result.dataset.has_datetime_index is True

    def test_id_column_detection(self, profiler: DataProfiler):
        df = pd.DataFrame({
            "user_id": range(50),
            "x": np.random.random(50),
        })
        result = profiler.profile(df)
        assert "user_id" in result.dataset.suspected_id_columns

    def test_leakage_by_name(self, profiler: DataProfiler):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "predicted_value": [1.1, 2.1, 3.1],
        })
        result = profiler.profile(df)
        leak_names = [r.column_name for r in result.dataset.data_leakage_risks]
        assert "predicted_value" in leak_names

    def test_high_correlation_detection(self, profiler: DataProfiler):
        x = np.arange(100, dtype=float)
        df = pd.DataFrame({"a": x, "b": x * 2 + 1, "c": np.random.random(100)})
        result = profiler.profile(df)
        pairs = result.dataset.high_correlation_pairs
        pair_cols = [(p.column_a, p.column_b) for p in pairs]
        assert ("a", "b") in pair_cols

    def test_duplicate_detection(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": [1, 2, 3, 1, 2, 3]})
        result = profiler.profile(df)
        assert result.dataset.duplicate_row_count == 3


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    """Verify profiler handles degenerate inputs gracefully."""

    def test_empty_dataframe(self, profiler: DataProfiler):
        df = pd.DataFrame()
        result = profiler.profile(df)
        assert len(result.columns) == 0
        assert result.dataset.num_rows == 0

    def test_single_row(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": [42], "y": ["hello"]})
        result = profiler.profile(df)
        assert len(result.columns) == 2
        assert result.dataset.num_rows == 1

    def test_all_null_dataframe(self, profiler: DataProfiler):
        df = pd.DataFrame({"a": [None, None], "b": [np.nan, np.nan]})
        result = profiler.profile(df)
        for cp in result.columns:
            assert cp.missing.missing_percentage == 100.0

    def test_profiler_version(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": [1]})
        result = profiler.profile(df)
        assert result.profiler_version == "0.1.0"

    def test_profiling_duration_positive(self, profiler: DataProfiler, basic_df: pd.DataFrame):
        result = profiler.profile(basic_df)
        assert result.profiling_duration_seconds > 0

    def test_sample_values_limited(self, profiler: DataProfiler):
        df = pd.DataFrame({"x": range(1000)})
        result = profiler.profile(df)
        assert len(result.columns[0].sample_values) <= 5
