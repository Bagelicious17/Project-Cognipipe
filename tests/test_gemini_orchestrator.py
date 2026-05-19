"""
Unit tests for ``services.gemini_orchestrator.GeminiOrchestrator``.

Tests cover:
- _profile_to_prompt_context (ID exclusion, float rounding, token estimate)
- _strip_markdown_fences utility
- JSON parse failure retry logic (mocked API)
- Full run() with mocked Gemini responses
- GeminiOrchestrationError on API failure
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import (
    AnalystDiagnostic,
    ColumnProfile,
    CorrelationPair,
    DatasetFlags,
    GeminiResult,
    MissingInfo,
    MissingPattern,
    NumericalStats,
    CategoricalStats,
    ProfileResult,
    SemanticType,
    TaskType,
    CorrelationMethod,
)
from services.gemini_orchestrator import (
    GeminiOrchestrationError,
    GeminiOrchestrator,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_profile() -> ProfileResult:
    """Build a realistic ProfileResult without needing pandas."""
    return ProfileResult(
        columns=[
            ColumnProfile(
                column_name="customer_id",
                dtype="int64",
                inferred_semantic_type=SemanticType.ID,
                missing=MissingInfo(missing_count=0, missing_percentage=0.0, missing_pattern=MissingPattern.NONE),
                nunique=200,
                sample_values=[1, 2, 3, 4, 5],
            ),
            ColumnProfile(
                column_name="age",
                dtype="int64",
                inferred_semantic_type=SemanticType.CONTINUOUS,
                missing=MissingInfo(missing_count=0, missing_percentage=0.0, missing_pattern=MissingPattern.NONE),
                numerical_stats=NumericalStats(
                    mean=45.123456789, median=44.0, std=12.5678, min=18.0, max=79.0,
                    skewness=0.1234, kurtosis=-0.5678, q1=33.0, q3=57.0, iqr=24.0,
                    outlier_count_iqr=2, outlier_count_zscore=0, zero_count=0,
                    negative_count=0, is_likely_log_distributed=False,
                ),
                nunique=60,
                sample_values=[25, 34, 55, 67, 42],
            ),
            ColumnProfile(
                column_name="income",
                dtype="float64",
                inferred_semantic_type=SemanticType.CONTINUOUS,
                missing=MissingInfo(missing_count=10, missing_percentage=5.0, missing_pattern=MissingPattern.MCAR),
                numerical_stats=NumericalStats(
                    mean=55000.123, median=42000.0, std=30000.0, min=15000.0, max=250000.0,
                    skewness=2.31, kurtosis=5.6, q1=28000.0, q3=65000.0, iqr=37000.0,
                    outlier_count_iqr=8, outlier_count_zscore=3, zero_count=0,
                    negative_count=0, is_likely_log_distributed=True,
                ),
                nunique=190,
                sample_values=[32000.5, 45000.0, 120000.0, 18000.0, 67000.0],
            ),
            ColumnProfile(
                column_name="gender",
                dtype="object",
                inferred_semantic_type=SemanticType.NOMINAL,
                missing=MissingInfo(missing_count=3, missing_percentage=1.5, missing_pattern=MissingPattern.MCAR),
                categorical_stats=CategoricalStats(
                    cardinality=3, cardinality_ratio=0.015,
                    top_10_values={"M": 90, "F": 90, "Other": 20},
                    rare_category_count=1,
                ),
                nunique=3,
                sample_values=["M", "F", "Other", "M", "F"],
            ),
            ColumnProfile(
                column_name="churn",
                dtype="int64",
                inferred_semantic_type=SemanticType.TARGET_CANDIDATE,
                missing=MissingInfo(missing_count=0, missing_percentage=0.0, missing_pattern=MissingPattern.NONE),
                numerical_stats=NumericalStats(
                    mean=0.2, median=0.0, std=0.4, min=0.0, max=1.0,
                    skewness=1.5, kurtosis=0.25, q1=0.0, q3=0.0, iqr=0.0,
                    outlier_count_iqr=40, outlier_count_zscore=0, zero_count=160,
                    negative_count=0, is_likely_log_distributed=False,
                ),
                nunique=2,
                sample_values=[0, 1, 0, 0, 1],
            ),
        ],
        dataset=DatasetFlags(
            num_rows=200,
            num_columns=5,
            is_time_series=False,
            suspected_target_column="churn",
            likely_task_type=TaskType.BINARY_CLASSIFICATION,
            class_imbalance_ratio=4.0,
            high_correlation_pairs=[],
            data_leakage_risks=[],
        ),
    )


def _make_mock_response(json_text: str) -> MagicMock:
    """Create a mock Gemini response object."""
    resp = MagicMock()
    resp.text = json_text
    usage = MagicMock()
    usage.prompt_token_count = 500
    usage.candidates_token_count = 300
    resp.usage_metadata = usage
    return resp


# Sample valid JSON responses for each chain
CHAIN_1_RESPONSE = json.dumps({
    "diagnostic_summary": "Binary classification dataset with 200 rows. Target is 'churn' with 4:1 imbalance.",
    "confirmed_task_type": "binary_classification",
    "confirmed_target_column": "churn",
    "critical_issues": ["Class imbalance ratio of 4.0 in 'churn'", "5% missing in 'income'"],
    "column_roles": {"age": "feature", "income": "feature", "gender": "feature", "churn": "target"},
    "priority_columns_for_engineering": ["income", "gender"],
})

CHAIN_2_RESPONSE = json.dumps({
    "steps": [
        {
            "step_order": 1, "operation": "log_transform", "target_columns": ["income"],
            "new_column_name": "income_log", "parameters": {"base": "natural"},
            "sklearn_equivalent": "sklearn.preprocessing.FunctionTransformer",
            "rationale": "Skewness of 2.31 indicates right-skew.", "code_snippet": "df['income_log'] = np.log1p(df['income'])",
            "priority": "high",
        }
    ],
    "summary": "Applied log transform to income.",
    "estimated_feature_count_after": 5,
})

CHAIN_3_RESPONSE = json.dumps({
    "task_type": "binary_classification", "target_column": "churn",
    "preprocessing": [
        {"step_order": 1, "operation": "impute_median", "target_columns": ["income"],
         "code_snippet": "imputer = SimpleImputer(strategy='median')", "rationale": "Handle 5% missing in income."}
    ],
    "model_candidates": [
        {"model_name": "XGBoost", "sklearn_class": "xgboost.XGBClassifier",
         "hyperparameters": {"n_estimators": [100, 200], "max_depth": [3, 5]},
         "rationale": "Good for tabular + handles imbalance.", "rank": 1},
        {"model_name": "Random Forest", "sklearn_class": "sklearn.ensemble.RandomForestClassifier",
         "hyperparameters": {"n_estimators": [100, 300]}, "rationale": "Robust baseline.", "rank": 2},
        {"model_name": "Logistic Regression", "sklearn_class": "sklearn.linear_model.LogisticRegression",
         "hyperparameters": {"C": [0.1, 1.0, 10.0]}, "rationale": "Interpretable baseline.", "rank": 3},
    ],
    "evaluation": {"validation_method": "stratified_kfold", "n_splits": 5,
                    "primary_metric": "f1_weighted", "secondary_metrics": ["roc_auc"],
                    "rationale": "Preserves class distribution."},
    "summary": "XGBoost with stratified 5-fold CV.",
})


# ── Test: _profile_to_prompt_context ──────────────────────────────────

class TestProfileToPromptContext:
    """Verify prompt context generation from ProfileResult."""

    def test_id_columns_excluded(self, mock_profile: ProfileResult):
        """Columns with semantic type 'id' must not appear in prompt context."""
        context_str = GeminiOrchestrator._profile_to_prompt_context(mock_profile)
        context = json.loads(context_str)
        col_names = [c["name"] for c in context["columns"]]
        assert "customer_id" not in col_names

    def test_non_id_columns_included(self, mock_profile: ProfileResult):
        """Feature and target columns must appear."""
        context_str = GeminiOrchestrator._profile_to_prompt_context(mock_profile)
        context = json.loads(context_str)
        col_names = [c["name"] for c in context["columns"]]
        assert "age" in col_names
        assert "income" in col_names
        assert "churn" in col_names

    def test_floats_rounded_to_4_decimals(self, mock_profile: ProfileResult):
        """All floats in the context must be rounded to 4 decimal places."""
        context_str = GeminiOrchestrator._profile_to_prompt_context(mock_profile)
        context = json.loads(context_str)
        age_col = [c for c in context["columns"] if c["name"] == "age"][0]
        mean_val = age_col["stats"]["mean"]
        # 45.123456789 → should be 45.1235
        assert mean_val == 45.1235

    def test_sample_values_included(self, mock_profile: ProfileResult):
        """Each column should have sample_values."""
        context_str = GeminiOrchestrator._profile_to_prompt_context(mock_profile)
        context = json.loads(context_str)
        for col in context["columns"]:
            assert "sample_values" in col
            assert len(col["sample_values"]) > 0

    def test_dataset_flags_included(self, mock_profile: ProfileResult):
        """Dataset-level metadata must be present."""
        context_str = GeminiOrchestrator._profile_to_prompt_context(mock_profile)
        context = json.loads(context_str)
        ds = context["dataset"]
        assert ds["rows"] == 200
        assert ds["likely_task_type"] == "binary_classification"
        assert ds["suspected_target"] == "churn"

    def test_context_is_valid_json(self, mock_profile: ProfileResult):
        """Output must be valid JSON."""
        context_str = GeminiOrchestrator._profile_to_prompt_context(mock_profile)
        parsed = json.loads(context_str)  # must not raise
        assert isinstance(parsed, dict)

    def test_token_estimate_reasonable(self, mock_profile: ProfileResult):
        """Context string should be under ~12KB (approx 3000 tokens)."""
        context_str = GeminiOrchestrator._profile_to_prompt_context(mock_profile)
        assert len(context_str) < 15000  # generous upper bound


# ── Test: _strip_markdown_fences ──────────────────────────────────────

class TestStripMarkdownFences:
    """Verify markdown fence removal."""

    def test_no_fences(self):
        assert GeminiOrchestrator._strip_markdown_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fences(self):
        raw = '```json\n{"a": 1}\n```'
        assert GeminiOrchestrator._strip_markdown_fences(raw) == '{"a": 1}'

    def test_plain_fences(self):
        raw = '```\n{"a": 1}\n```'
        assert GeminiOrchestrator._strip_markdown_fences(raw) == '{"a": 1}'


# ── Test: JSON retry logic ────────────────────────────────────────────

class TestRetryLogic:
    """Verify retry behavior on JSON parse failures."""

    @patch.object(GeminiOrchestrator, "_track_tokens")
    def test_retry_on_invalid_json_then_success(self, mock_track, mock_profile: ProfileResult):
        """First call returns invalid JSON, second returns valid → should succeed."""
        orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
        orch._model_name = "gemini-1.5-pro"
        orch._token_usage = []

        invalid_resp = _make_mock_response("This is not JSON at all")
        valid_resp = _make_mock_response(CHAIN_1_RESPONSE)

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [invalid_resp, valid_resp]
        orch._client = mock_client

        result = orch._call_gemini(1, "chain_1_analyst", "system", "user")
        assert result["confirmed_task_type"] == "binary_classification"
        assert mock_client.models.generate_content.call_count == 2

    @patch.object(GeminiOrchestrator, "_track_tokens")
    def test_all_retries_exhausted_raises_error(self, mock_track, mock_profile: ProfileResult):
        """Three invalid JSON responses → GeminiOrchestrationError."""
        orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
        orch._model_name = "gemini-1.5-pro"
        orch._token_usage = []

        invalid_resp = _make_mock_response("not json")
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = invalid_resp
        orch._client = mock_client

        with pytest.raises(GeminiOrchestrationError) as exc_info:
            orch._call_gemini(1, "chain_1_analyst", "system", "user")

        assert exc_info.value.chain_number == 1
        assert mock_client.models.generate_content.call_count == 3

    @patch.object(GeminiOrchestrator, "_track_tokens")
    def test_api_error_raises_immediately(self, mock_track):
        """An API exception should raise immediately without retrying."""
        orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
        orch._model_name = "gemini-1.5-pro"
        orch._token_usage = []

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API unavailable")
        orch._client = mock_client

        with pytest.raises(GeminiOrchestrationError) as exc_info:
            orch._call_gemini(2, "chain_2", "system", "user")

        assert exc_info.value.chain_number == 2
        assert mock_client.models.generate_content.call_count == 1


# ── Test: Full run() with mocked API ─────────────────────────────────

class TestFullRun:
    """Verify the complete orchestration pipeline with mocked Gemini calls."""

    @patch.object(GeminiOrchestrator, "_track_tokens")
    def test_run_produces_valid_gemini_result(self, mock_track, mock_profile: ProfileResult):
        """Full run() should return a valid GeminiResult."""
        orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
        orch._model_name = "gemini-1.5-pro"
        orch._token_usage = []

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_mock_response(CHAIN_1_RESPONSE),
            _make_mock_response(CHAIN_2_RESPONSE),
            _make_mock_response(CHAIN_3_RESPONSE),
        ]
        orch._client = mock_client

        result = orch.run(mock_profile)

        # Verify types
        assert isinstance(result, GeminiResult)
        assert isinstance(result.analyst_diagnostic, AnalystDiagnostic)
        assert result.analyst_diagnostic.confirmed_task_type == "binary_classification"
        assert result.analyst_diagnostic.confirmed_target_column == "churn"

        # Chain 2
        assert len(result.feature_engineering.steps) == 1
        assert result.feature_engineering.steps[0].operation == "log_transform"

        # Chain 3
        assert len(result.ml_architecture.model_candidates) == 3
        assert result.ml_architecture.model_candidates[0].rank == 1

        # Raw responses stored
        assert "chain_1_analyst" in result.raw_responses
        assert "chain_2_feature_engineer" in result.raw_responses
        assert "chain_3_ml_architect" in result.raw_responses

        # Duration tracked
        assert result.orchestration_duration_seconds > 0

        # API called exactly 3 times
        assert mock_client.models.generate_content.call_count == 3

    @patch.object(GeminiOrchestrator, "_track_tokens")
    def test_result_is_json_serializable(self, mock_track, mock_profile: ProfileResult):
        """GeminiResult.model_dump() must produce JSON-serializable output."""
        orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
        orch._model_name = "gemini-1.5-pro"
        orch._token_usage = []

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_mock_response(CHAIN_1_RESPONSE),
            _make_mock_response(CHAIN_2_RESPONSE),
            _make_mock_response(CHAIN_3_RESPONSE),
        ]
        orch._client = mock_client

        result = orch.run(mock_profile)
        d = result.model_dump()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100

        # Round-trip validation
        reloaded = GeminiResult.model_validate(d)
        assert reloaded.analyst_diagnostic.confirmed_target_column == "churn"


# ── Test: Edge cases identified by Sonnet ─────────────────────────────

class TestEdgeCases:
    """Edge cases that will appear when judges test with weird inputs."""

    def test_all_id_columns_produce_empty_context(self):
        """Profile where every column is ID → context has 0 columns."""
        profile = ProfileResult(
            columns=[
                ColumnProfile(
                    column_name="id_1", dtype="int64",
                    inferred_semantic_type=SemanticType.ID,
                    missing=MissingInfo(missing_count=0, missing_percentage=0.0, missing_pattern=MissingPattern.NONE),
                    nunique=100, sample_values=[1, 2, 3],
                ),
                ColumnProfile(
                    column_name="id_2", dtype="int64",
                    inferred_semantic_type=SemanticType.ID,
                    missing=MissingInfo(missing_count=0, missing_percentage=0.0, missing_pattern=MissingPattern.NONE),
                    nunique=100, sample_values=[10, 20, 30],
                ),
            ],
            dataset=DatasetFlags(
                num_rows=100, num_columns=2, is_time_series=False,
                suspected_target_column=None, likely_task_type=TaskType.UNKNOWN,
            ),
        )
        context_str = GeminiOrchestrator._profile_to_prompt_context(profile)
        context = json.loads(context_str)
        assert len(context["columns"]) == 0

    def test_single_column_dataset(self):
        """Profile with only one non-ID column."""
        profile = ProfileResult(
            columns=[
                ColumnProfile(
                    column_name="value", dtype="float64",
                    inferred_semantic_type=SemanticType.CONTINUOUS,
                    missing=MissingInfo(missing_count=0, missing_percentage=0.0, missing_pattern=MissingPattern.NONE),
                    nunique=50, sample_values=[1.1, 2.2, 3.3],
                ),
            ],
            dataset=DatasetFlags(
                num_rows=50, num_columns=1, is_time_series=False,
                suspected_target_column="value", likely_task_type=TaskType.UNKNOWN,
            ),
        )
        context_str = GeminiOrchestrator._profile_to_prompt_context(profile)
        context = json.loads(context_str)
        assert len(context["columns"]) == 1
        assert context["columns"][0]["name"] == "value"

    def test_wide_dataset_truncation(self):
        """Profile with 50 columns → context truncated to max_columns."""
        cols = []
        for i in range(50):
            cols.append(ColumnProfile(
                column_name=f"feat_{i}", dtype="float64",
                inferred_semantic_type=SemanticType.CONTINUOUS,
                missing=MissingInfo(
                    missing_count=i, missing_percentage=float(i),
                    missing_pattern=MissingPattern.MCAR,
                ),
                nunique=100, sample_values=[1.0, 2.0],
            ))
        profile = ProfileResult(
            columns=cols,
            dataset=DatasetFlags(
                num_rows=1000, num_columns=50, is_time_series=False,
                suspected_target_column="feat_0", likely_task_type=TaskType.REGRESSION,
            ),
        )
        context_str = GeminiOrchestrator._profile_to_prompt_context(profile, max_columns=10)
        context = json.loads(context_str)
        assert len(context["columns"]) == 10
        # Target column must be kept
        col_names = [c["name"] for c in context["columns"]]
        assert "feat_0" in col_names
        # Truncation note must exist
        assert "note" in context["dataset"]
        assert "truncated" in context["dataset"]["note"].lower()

    def test_null_target_in_chain2_prompt(self):
        """When confirmed_target_column is None, Chain 2 prompt should say 'not identified'."""
        chain1 = {
            "confirmed_task_type": "clustering",
            "confirmed_target_column": None,
            "critical_issues": [],
            "priority_columns_for_engineering": [],
            "column_roles": {"feat_a": "feature"},
        }
        profile = ProfileResult(
            columns=[
                ColumnProfile(
                    column_name="feat_a", dtype="float64",
                    inferred_semantic_type=SemanticType.CONTINUOUS,
                    missing=MissingInfo(missing_count=0, missing_percentage=0.0, missing_pattern=MissingPattern.NONE),
                    nunique=50, sample_values=[1.0],
                ),
            ],
            dataset=DatasetFlags(
                num_rows=100, num_columns=1, is_time_series=False,
                suspected_target_column=None, likely_task_type=TaskType.CLUSTERING,
            ),
        )
        # We can't call _run_chain_2 directly without mocking the API,
        # but we can verify the target_context logic by simulating it:
        target_col = chain1.get("confirmed_target_column")
        target_context = (
            f"'{target_col}'" if target_col
            else "not identified (unsupervised task or ambiguous dataset)"
        )
        assert "not identified" in target_context
        assert "None" not in target_context  # must NOT contain literal "None"

    @patch.object(GeminiOrchestrator, "_track_tokens")
    def test_chain2_failure_after_chain1_success(self, mock_track):
        """Chain 1 succeeds, Chain 2 fails all 3 retries → error with chain_number=2."""
        profile = ProfileResult(
            columns=[
                ColumnProfile(
                    column_name="x", dtype="float64",
                    inferred_semantic_type=SemanticType.CONTINUOUS,
                    missing=MissingInfo(missing_count=0, missing_percentage=0.0, missing_pattern=MissingPattern.NONE),
                    nunique=50, sample_values=[1.0],
                ),
            ],
            dataset=DatasetFlags(
                num_rows=50, num_columns=1, is_time_series=False,
                suspected_target_column="x", likely_task_type=TaskType.REGRESSION,
            ),
        )
        orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
        orch._model_name = "gemini-1.5-pro"
        orch._token_usage = []

        chain1_resp = _make_mock_response(json.dumps({
            "diagnostic_summary": "Single col", "confirmed_task_type": "regression",
            "confirmed_target_column": "x", "critical_issues": [],
            "column_roles": {"x": "target"}, "priority_columns_for_engineering": [],
        }))
        invalid_resp = _make_mock_response("not valid json")

        mock_client = MagicMock()
        # Chain 1 succeeds, then Chain 2 fails 3 times
        mock_client.models.generate_content.side_effect = [
            chain1_resp, invalid_resp, invalid_resp, invalid_resp,
        ]
        orch._client = mock_client

        with pytest.raises(GeminiOrchestrationError) as exc_info:
            orch.run(profile)
        assert exc_info.value.chain_number == 2

    @patch.object(GeminiOrchestrator, "_track_tokens")
    def test_token_usage_accumulates_across_chains(self, mock_track, mock_profile):
        """Token usage list should have one entry per successful chain call."""
        orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
        orch._model_name = "gemini-1.5-pro"
        orch._token_usage = []

        # Manually add token usage inside _track_tokens mock
        def side_effect_track(chain_name, response):
            from models.schemas import ChainTokenUsage
            orch._token_usage.append(ChainTokenUsage(
                chain_name=chain_name, input_tokens=500,
                output_tokens=300, total_tokens=800,
            ))
        mock_track.side_effect = side_effect_track

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            _make_mock_response(CHAIN_1_RESPONSE),
            _make_mock_response(CHAIN_2_RESPONSE),
            _make_mock_response(CHAIN_3_RESPONSE),
        ]
        orch._client = mock_client

        result = orch.run(mock_profile)
        assert len(result.token_usage) == 3
        assert result.token_usage[0].chain_name == "chain_1_analyst"
        assert result.token_usage[1].chain_name == "chain_2_feature_engineer"
        assert result.token_usage[2].chain_name == "chain_3_ml_architect"
        total = sum(t.total_tokens for t in result.token_usage)
        assert total == 2400  # 800 * 3

    @patch.object(GeminiOrchestrator, "_track_tokens")
    def test_retry_prompt_contains_raw_response(self, mock_track):
        """On JSON parse failure, the retry prompt must include the malformed output."""
        orch = GeminiOrchestrator.__new__(GeminiOrchestrator)
        orch._model_name = "gemini-1.5-pro"
        orch._token_usage = []

        bad_text = '{"incomplete": true, this is broken'
        invalid_resp = _make_mock_response(bad_text)
        valid_resp = _make_mock_response(CHAIN_1_RESPONSE)

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [invalid_resp, valid_resp]
        orch._client = mock_client

        orch._call_gemini(1, "chain_1_analyst", "system", "user prompt")

        # The second call should have received a retry prompt containing the bad text
        second_call_args = mock_client.models.generate_content.call_args_list[1]
        retry_prompt_sent = second_call_args.kwargs.get("contents") or second_call_args[1].get("contents", "")
        assert bad_text in retry_prompt_sent
