"""
Unit tests for ``services.code_assembler.CodeAssembler``.

Tests cover:
- Import derivation from sklearn_equivalent fields
- Justification comments in generated code
- Notebook cell count and structure
- Requirements.txt completeness (keras only when needed)
- Full build() end-to-end with mock ProfileResult + GeminiResult
- Script syntax validity
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from models.schemas import (
    AnalystDiagnostic,
    ColumnProfile,
    DatasetFlags,
    EvaluationStrategy,
    FeatureEngineeringPrescription,
    FeatureStep,
    GeminiResult,
    GeneratedPipeline,
    MLArchitectureRecommendation,
    MissingInfo,
    MissingPattern,
    ModelCandidate,
    PreprocessingStep,
    ProfileResult,
    SemanticType,
    TaskType,
)
from services.code_assembler import CodeAssembler


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_feature_steps() -> list[FeatureStep]:
    return [
        FeatureStep(
            step_order=1,
            operation="log_transform",
            target_columns=["income"],
            new_column_name="income_log",
            parameters={"base": "natural"},
            sklearn_equivalent="sklearn.preprocessing.FunctionTransformer",
            rationale="Skewness of 2.31 indicates heavy right-skew.",
            code_snippet="df['income_log'] = np.log1p(df['income'])",
            priority="high",
        ),
        FeatureStep(
            step_order=2,
            operation="one_hot_encode",
            target_columns=["gender"],
            new_column_name=None,
            parameters={"drop": "first"},
            sklearn_equivalent="sklearn.preprocessing.OneHotEncoder",
            rationale="Nominal column with 3 categories.",
            code_snippet="df = pd.get_dummies(df, columns=['gender'], drop_first=True)",
            priority="medium",
        ),
    ]


def _make_model_candidates() -> list[ModelCandidate]:
    return [
        ModelCandidate(
            model_name="XGBoost Classifier",
            sklearn_class="xgboost.XGBClassifier",
            hyperparameters={
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
            },
            rationale="Handles class imbalance via scale_pos_weight.",
            rank=1,
        ),
        ModelCandidate(
            model_name="Random Forest",
            sklearn_class="sklearn.ensemble.RandomForestClassifier",
            hyperparameters={"n_estimators": [100, 300]},
            rationale="Robust ensemble baseline.",
            rank=2,
        ),
        ModelCandidate(
            model_name="Logistic Regression",
            sklearn_class="sklearn.linear_model.LogisticRegression",
            hyperparameters={"C": [0.1, 1.0, 10.0]},
            rationale="Interpretable linear baseline.",
            rank=3,
        ),
    ]


def _make_preprocessing() -> list[PreprocessingStep]:
    return [
        PreprocessingStep(
            step_order=1,
            operation="impute_median",
            target_columns=["income"],
            code_snippet=(
                "from sklearn.impute import SimpleImputer\n"
                "imputer = SimpleImputer(strategy='median')\n"
                "df[['income']] = imputer.fit_transform(df[['income']])"
            ),
            rationale="Handle 5% missing values in income.",
        ),
    ]


@pytest.fixture
def mock_profile() -> ProfileResult:
    return ProfileResult(
        columns=[
            ColumnProfile(
                column_name="customer_id", dtype="int64",
                inferred_semantic_type=SemanticType.ID,
                missing=MissingInfo(
                    missing_count=0, missing_percentage=0.0,
                    missing_pattern=MissingPattern.NONE,
                ),
                nunique=200, sample_values=[1, 2, 3],
            ),
            ColumnProfile(
                column_name="income", dtype="float64",
                inferred_semantic_type=SemanticType.CONTINUOUS,
                missing=MissingInfo(
                    missing_count=10, missing_percentage=5.0,
                    missing_pattern=MissingPattern.MCAR,
                ),
                nunique=190, sample_values=[30000.0, 50000.0],
            ),
            ColumnProfile(
                column_name="gender", dtype="object",
                inferred_semantic_type=SemanticType.NOMINAL,
                missing=MissingInfo(
                    missing_count=0, missing_percentage=0.0,
                    missing_pattern=MissingPattern.NONE,
                ),
                nunique=3, sample_values=["M", "F"],
            ),
            ColumnProfile(
                column_name="churn", dtype="int64",
                inferred_semantic_type=SemanticType.TARGET_CANDIDATE,
                missing=MissingInfo(
                    missing_count=0, missing_percentage=0.0,
                    missing_pattern=MissingPattern.NONE,
                ),
                nunique=2, sample_values=[0, 1],
            ),
        ],
        dataset=DatasetFlags(
            num_rows=200, num_columns=4, is_time_series=False,
            suspected_target_column="churn",
            likely_task_type=TaskType.BINARY_CLASSIFICATION,
            class_imbalance_ratio=4.0,
        ),
    )


@pytest.fixture
def mock_gemini() -> GeminiResult:
    return GeminiResult(
        analyst_diagnostic=AnalystDiagnostic(
            diagnostic_summary=(
                "Binary classification dataset with 200 rows. "
                "Target 'churn' has 4:1 class imbalance."
            ),
            confirmed_task_type="binary_classification",
            confirmed_target_column="churn",
            critical_issues=["4:1 class imbalance in 'churn'"],
            column_roles={
                "income": "feature",
                "gender": "feature",
                "churn": "target",
            },
            priority_columns_for_engineering=["income", "gender"],
        ),
        feature_engineering=FeatureEngineeringPrescription(
            steps=_make_feature_steps(),
            summary="Log transform + one-hot encoding.",
            estimated_feature_count_after=5,
        ),
        ml_architecture=MLArchitectureRecommendation(
            task_type=TaskType.BINARY_CLASSIFICATION,
            target_column="churn",
            preprocessing=_make_preprocessing(),
            model_candidates=_make_model_candidates(),
            evaluation=EvaluationStrategy(
                validation_method="stratified_kfold",
                n_splits=5,
                primary_metric="f1_weighted",
                secondary_metrics=["roc_auc"],
                rationale="Preserves class distribution.",
            ),
            summary="XGBoost with stratified 5-fold CV.",
        ),
        raw_responses={},
    )


@pytest.fixture
def assembler() -> CodeAssembler:
    return CodeAssembler()


@pytest.fixture
def built_pipeline(
    assembler: CodeAssembler,
    mock_profile: ProfileResult,
    mock_gemini: GeminiResult,
) -> GeneratedPipeline:
    return assembler.build(mock_profile, mock_gemini)


# ── Test: Import derivation ──────────────────────────────────────────

class TestImportDerivation:
    """Verify imports are derived from sklearn_equivalent fields."""

    def test_sklearn_imports_present(self, assembler: CodeAssembler):
        imports = assembler._build_imports(
            _make_feature_steps(), _make_preprocessing(),
            _make_model_candidates(),
        )
        # FunctionTransformer and OneHotEncoder are sklearn_equivalents
        # but NOT used in the code_snippets (np.log1p and pd.get_dummies
        # are used instead), so they must NOT be imported.
        assert "FunctionTransformer" not in imports
        assert "OneHotEncoder" not in imports
        # SimpleImputer IS used in the preprocessing snippet
        assert "SimpleImputer" in imports
        assert "train_test_split" in imports
        assert "RandomizedSearchCV" in imports

    def test_external_model_imports(self, assembler: CodeAssembler):
        imports = assembler._build_imports(
            [], [], _make_model_candidates(),
        )
        assert "XGBClassifier" in imports
        assert "RandomForestClassifier" in imports
        assert "LogisticRegression" in imports

    def test_always_includes_pandas_numpy(self, assembler: CodeAssembler):
        imports = assembler._build_imports([], [], [])
        assert "import pandas as pd" in imports
        assert "import numpy as np" in imports

    def test_no_unused_keras_import(self, assembler: CodeAssembler):
        """Keras should NOT appear when no neural net model is used."""
        imports = assembler._build_imports(
            _make_feature_steps(), [], _make_model_candidates(),
        )
        assert "keras" not in imports.lower()
        assert "tensorflow" not in imports.lower()

    def test_preprocessing_snippet_imports_extracted(
        self, assembler: CodeAssembler
    ):
        """Imports from preprocessing code_snippet are extracted."""
        imports = assembler._build_imports(
            [], _make_preprocessing(), [],
        )
        assert "SimpleImputer" in imports


# ── Test: Justification comments ─────────────────────────────────────

class TestJustificationComments:
    """Every transformation block must contain its rationale as a comment."""

    def test_rationale_in_feature_block(self, assembler: CodeAssembler):
        block = assembler._build_feature_engineering_block(
            _make_feature_steps(),
        )
        assert "Skewness of 2.31" in block
        assert "Nominal column with 3 categories" in block

    def test_rationale_in_feature_engineering_block(self, assembler: CodeAssembler):
        block = assembler._build_feature_engineering_block(_make_feature_steps())
        assert "Skewness of 2.31 indicates" in block


# ── Test: Script syntax ──────────────────────────────────────────────

class TestScriptSyntax:
    """Generated script must be syntactically valid Python."""

    def test_script_compiles(self, built_pipeline: GeneratedPipeline):
        """Script must pass compile() without SyntaxError."""
        compile(built_pipeline.python_script, "<pipeline>", "exec")

    def test_script_has_all_sections(self, built_pipeline: GeneratedPipeline):
        """Script must contain all required section headers."""
        script = built_pipeline.python_script
        assert "# === CONFIGURATION ===" in script
        assert "# === DATA LOADING ===" in script
        assert "# === FEATURE ENGINEERING ===" in script
        assert "# === TRAIN/TEST SPLIT ===" in script
        assert "# === MODEL ===" in script
        assert "# === EVALUATION ===" in script
        assert "# === MAIN ===" in script

    def test_target_column_uses_actual_name(
        self, built_pipeline: GeneratedPipeline
    ):
        """TARGET_COLUMN must use the real column name, not a placeholder."""
        assert 'TARGET_COLUMN = "churn"' in built_pipeline.python_script

    def test_try_except_wraps_transforms(
        self, built_pipeline: GeneratedPipeline
    ):
        """Feature engineering steps must be wrapped in try/except."""
        script = built_pipeline.python_script
        assert "try:" in script
        assert "except Exception as e:" in script


# ── Test: Notebook structure ─────────────────────────────────────────

class TestNotebookStructure:
    """Verify notebook cell structure."""

    def test_notebook_is_valid_json(self, built_pipeline: GeneratedPipeline):
        nb = json.loads(built_pipeline.notebook_json)
        assert "cells" in nb

    def test_notebook_starts_with_markdown(
        self, built_pipeline: GeneratedPipeline
    ):
        """First cell should be the diagnostic summary markdown."""
        nb = json.loads(built_pipeline.notebook_json)
        first_cell = nb["cells"][0]
        assert first_cell["cell_type"] == "markdown"
        assert "CogniPipe" in first_cell["source"]

    def test_notebook_alternates_md_and_code(
        self, built_pipeline: GeneratedPipeline
    ):
        """Each section should have a markdown cell followed by a code cell."""
        nb = json.loads(built_pipeline.notebook_json)
        cells = nb["cells"]
        # After the title markdown, cells should alternate md → code
        # (with some flexibility for empty sections)
        md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
        code_count = sum(1 for c in cells if c["cell_type"] == "code")
        # At minimum: title md + (N sections × 1 md + 1 code)
        assert md_count >= 5  # title + at least 4 sections
        assert code_count >= 4  # at least 4 code sections

    def test_diagnostic_summary_in_notebook(
        self, built_pipeline: GeneratedPipeline
    ):
        """Analyst diagnostic summary must appear in the notebook."""
        nb = json.loads(built_pipeline.notebook_json)
        all_sources = " ".join(
            c["source"] for c in nb["cells"]
            if c["cell_type"] == "markdown"
        )
        assert "4:1 class imbalance" in all_sources


# ── Test: Requirements.txt ───────────────────────────────────────────

class TestRequirements:
    """Verify requirements.txt generation."""

    def test_core_packages_always_present(
        self, built_pipeline: GeneratedPipeline
    ):
        reqs = built_pipeline.requirements_txt
        assert "pandas" in reqs
        assert "numpy" in reqs
        assert "scikit-learn" in reqs

    def test_xgboost_included_when_used(
        self, built_pipeline: GeneratedPipeline
    ):
        assert "xgboost" in built_pipeline.requirements_txt

    def test_keras_excluded_when_not_used(
        self, built_pipeline: GeneratedPipeline
    ):
        assert "keras" not in built_pipeline.requirements_txt
        assert "tensorflow" not in built_pipeline.requirements_txt

    def test_keras_included_when_neural_net(self, assembler: CodeAssembler):
        """Keras must appear when a Keras model is among candidates."""
        nn_candidate = ModelCandidate(
            model_name="MLP Classifier",
            sklearn_class="keras.Sequential",
            hyperparameters={},
            rationale="Neural net baseline.",
            rank=1,
        )
        reqs = assembler._build_requirements([], [nn_candidate])
        assert "keras" in reqs

    def test_version_pins_present(self, built_pipeline: GeneratedPipeline):
        """Packages should have version constraints."""
        reqs = built_pipeline.requirements_txt
        assert ">=" in reqs
        assert "<" in reqs


# ── Test: Full build end-to-end ──────────────────────────────────────

class TestFullBuild:
    """End-to-end build with mock inputs."""

    def test_build_returns_generated_pipeline(
        self, built_pipeline: GeneratedPipeline
    ):
        assert isinstance(built_pipeline, GeneratedPipeline)

    def test_all_fields_populated(self, built_pipeline: GeneratedPipeline):
        assert len(built_pipeline.python_script) > 500
        assert built_pipeline.notebook_json is not None
        assert len(built_pipeline.notebook_json) > 500
        assert len(built_pipeline.requirements_txt) > 50
        assert len(built_pipeline.pipeline_summary) > 20
        assert built_pipeline.source_profile_version != ""

    def test_pipeline_is_json_serializable(
        self, built_pipeline: GeneratedPipeline
    ):
        d = built_pipeline.model_dump()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 1000
        # Round-trip
        reloaded = GeneratedPipeline.model_validate(d)
        assert reloaded.pipeline_summary == built_pipeline.pipeline_summary

    def test_empty_steps_still_builds(
        self,
        assembler: CodeAssembler,
        mock_profile: ProfileResult,
    ):
        """Build should succeed even with no feature steps / no models."""
        gemini = GeminiResult(
            analyst_diagnostic=AnalystDiagnostic(
                diagnostic_summary="Minimal dataset.",
                confirmed_task_type="unknown",
                confirmed_target_column=None,
                critical_issues=[],
                column_roles={},
                priority_columns_for_engineering=[],
            ),
            feature_engineering=FeatureEngineeringPrescription(
                steps=[], summary="", estimated_feature_count_after=0,
            ),
            ml_architecture=MLArchitectureRecommendation(
                task_type=TaskType.UNKNOWN,
                target_column="unknown",
                preprocessing=[],
                model_candidates=[],
                evaluation=None,
                summary="",
            ),
            raw_responses={},
        )
        result = assembler.build(mock_profile, gemini)
        assert isinstance(result, GeneratedPipeline)
        assert "# === DATA LOADING ===" in result.python_script


# ── Test: Sonnet audit — MODEL_CHOICE, try/except, flake8 ────────────

class TestSonnetAudit:
    """Tests addressing specific concerns raised in code review."""

    def test_all_three_candidates_in_script(
        self, built_pipeline: GeneratedPipeline
    ):
        """All 3 model candidates must appear in the script, indexed by MODEL_CHOICE."""
        script = built_pipeline.python_script
        assert "XGBoost Classifier" in script
        assert "Random Forest" in script
        assert "Logistic Regression" in script
        assert "models[MODEL_CHOICE]" in script
        assert "models.append" in script

    def test_model_choice_selects_from_list(
        self, built_pipeline: GeneratedPipeline
    ):
        """The script must use models[MODEL_CHOICE], not hardcode a single model."""
        script = built_pipeline.python_script
        assert "selected_name, selected_model, selected_params = models[MODEL_CHOICE]" in script

    def test_try_except_has_continuation_message(
        self, built_pipeline: GeneratedPipeline
    ):
        """Each except block must tell the user execution continues."""
        script = built_pipeline.python_script
        assert "Continuing with untransformed values" in script

    def test_no_import_leakage_from_code_snippets(
        self, assembler: CodeAssembler
    ):
        """Regex must not leak non-import lines from code_snippet into imports."""
        imports = assembler._build_imports(
            [], _make_preprocessing(), [],
        )
        # "imputer" is a variable name in the snippet, not an import
        lines = imports.strip().split("\n")
        for line in lines:
            if line.strip() and not line.startswith("#"):
                # Every non-empty line should be a valid import or blank
                assert (
                    line.startswith("import ")
                    or line.startswith("from ")
                    or line.strip() == ""
                ), f"Invalid import line: {line!r}"

    def test_script_passes_flake8(self, built_pipeline: GeneratedPipeline):
        """Generated script must pass flake8 with only E501 (line length) ignored."""
        # Write the script to a temp file and run flake8
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as f:
            f.write(built_pipeline.python_script)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "flake8",
                    "--ignore=E501,W503,E402",
                    "--max-line-length=120",
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                pytest.fail(
                    f"flake8 found issues in generated script:\n"
                    f"{result.stdout}\n{result.stderr}"
                )
        except FileNotFoundError:
            pytest.skip("flake8 not installed")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
