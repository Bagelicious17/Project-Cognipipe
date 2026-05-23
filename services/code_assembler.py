"""
CogniPipe — Layer 3: CodeAssembler
====================================

Pure string-assembly layer.  Takes a ``ProfileResult`` (Layer 1) and a
``GeminiResult`` (Layer 2) and produces a self-contained, runnable
Python pipeline script, a Jupyter Notebook, and a requirements.txt.

Rules:
- No pandas in this file.  Ever.
- No AI calls.  Ever.
- Only string manipulation and ``nbformat`` for notebook generation.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import nbformat

import pprint

from models.schemas import (
    FeatureEngineeringPrescription,
    FeatureStep,
    GeminiResult,
    MLArchitectureRecommendation,
    ModelCandidate,
    PreprocessingStep,
    ProfileResult,
    GeneratedPipeline,
    TaskType,
)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

_VERSION = "0.1.0"

# Maps top-level Python package names to pip package names.
_PACKAGE_MAP: dict[str, str] = {
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "keras": "keras",
    "tensorflow": "tensorflow",
    "imblearn": "imbalanced-learn",
}

# Pinned version ranges for deterministic environments.
_VERSION_PINS: dict[str, str] = {
    "pandas": "pandas>=2.0.0,<3.0.0",
    "numpy": "numpy>=1.24.0,<3.0.0",
    "scikit-learn": "scikit-learn>=1.3.0,<2.0.0",
    "xgboost": "xgboost>=2.0.0,<3.0.0",
    "lightgbm": "lightgbm>=4.0.0,<5.0.0",
    "catboost": "catboost>=1.2.0,<2.0.0",
    "keras": "keras>=3.0.0,<4.0.0",
    "tensorflow": "tensorflow>=2.15.0,<3.0.0",
    "imbalanced-learn": "imbalanced-learn>=0.12.0,<1.0.0",
    "matplotlib": "matplotlib>=3.7.0,<4.0.0",
}


# ──────────────────────────────────────────────────────────────────────
# CodeAssembler
# ──────────────────────────────────────────────────────────────────────


class CodeAssembler:
    """Assembles a production-ready ML pipeline from profiling and AI outputs.

    Usage::

        assembler = CodeAssembler()
        pipeline = assembler.build(profile_result, gemini_result)
        # pipeline.python_script  → str
        # pipeline.notebook_json  → str (nbformat JSON)
        # pipeline.requirements_txt → str
    """

    def build(
        self,
        profile: ProfileResult,
        gemini: GeminiResult,
    ) -> GeneratedPipeline:
        """Build the complete pipeline output.

        Args:
            profile: Layer 1 profiling result.
            gemini: Layer 2 AI orchestration result.

        Returns:
            ``GeneratedPipeline`` with script, notebook, and requirements.
        """
        fe = gemini.feature_engineering
        ml = gemini.ml_architecture
        analyst = gemini.analyst_diagnostic

        # Pick the top-ranked model candidate
        candidates = sorted(ml.model_candidates, key=lambda c: c.rank)
        primary_candidate = candidates[0] if candidates else None

        # Assemble the script sections
        docstring = self._build_docstring(profile, gemini)
        imports = self._build_imports(
            fe.steps, ml.preprocessing, candidates,
        )
        config = self._build_config(gemini)
        loading = self._build_loading_block(profile)
        feature_eng = self._build_feature_engineering_block(fe.steps)

        # Remove duplicate blocks: The ML Architecture 'preprocessing' step is skipped 
        # as it is a duplicate of Feature Engineering. The preprocessing section is dropped entirely.
        # Assemble the body first to filter unused imports
        split = self._build_split_block(ml)
        model_block = self._build_model_block(candidates)
        evaluation = self._build_evaluation_block(ml)
        main_block = self._build_main_block()

        script_body = self._assemble_script(
            config, loading, feature_eng, split,
            model_block, evaluation, main_block,
        )

        imports = self._build_imports(
            fe.steps, ml.preprocessing, candidates, script_body
        )

        script = self._assemble_script(
            docstring, imports, script_body,
        )

        notebook_json = self._build_notebook(script, gemini)
        requirements = self._build_requirements(fe.steps, candidates)

        summary = (
            analyst.diagnostic_summary
            if analyst.diagnostic_summary
            else "Auto-generated ML pipeline."
        )

        return GeneratedPipeline(
            python_script=script,
            notebook_json=notebook_json,
            requirements_txt=requirements,
            pipeline_summary=summary,
            source_profile_version=profile.profiler_version,
        )

    # ── Import derivation ─────────────────────────────────────────

    def _build_imports(
        self,
        steps: list[FeatureStep],
        preprocessing: list[PreprocessingStep],
        candidates: list[ModelCandidate],
        script_body: str = "",
    ) -> str:
        """Derive import statements from actual sklearn_equivalent fields.

        Import groups (PEP 8 order):
        1. Standard library (omitted if nothing needed)
        2. Third-party (pandas, numpy)
        3. sklearn modules
        4. External ML libs (xgboost, lightgbm, keras)

        Only imports classes that actually appear in the generated code
        snippets to avoid flake8 F401 (unused import) violations.
        """
        # Collect all fully-qualified class paths that are actually USED
        # in the generated code output.
        class_paths: set[str] = set()
        std_imports: set[str] = set()

        for step in steps:
            if step.sklearn_equivalent:
                # Only import if the class name actually appears in the
                # code_snippet (e.g. skip FunctionTransformer when the
                # snippet uses np.log1p directly).
                cls_name = step.sklearn_equivalent.rsplit(".", 1)[-1]
                if cls_name in step.code_snippet:
                    class_paths.add(step.sklearn_equivalent)
            
            # Extract plain imports
            for match in re.findall(r"^import\s+([a-z_][a-z0-9_]*)", step.code_snippet, re.MULTILINE):
                if f"{match}." in step.code_snippet:
                    std_imports.add(match)

        for step in preprocessing:
            # Extract plain imports
            for match in re.findall(r"^import\s+([a-z_][a-z0-9_]*)", step.code_snippet, re.MULTILINE):
                if f"{match}." in step.code_snippet:
                    std_imports.add(match)

            # Extract "from X import Y" lines from code_snippet.
            # These classes are hoisted to top-level imports and the
            # inline import line is stripped from the code block.
            for match in re.findall(
                r"^from\s+([\w.]+)\s+import\s+([A-Z]\w*(?:\s*,\s*[A-Z]\w*)*)",
                step.code_snippet,
                re.MULTILINE,
            ):
                module, names = match
                for name in names.split(","):
                    name = name.strip()
                    if name:
                        class_paths.add(f"{module}.{name}")

        for cand in candidates:
            if cand.sklearn_class:
                class_paths.add(cand.sklearn_class)

        # Filter out unused imports based on script_body
        if script_body:
            script_body_no_comments = re.sub(r"#.*", "", script_body)
            used_paths = set()
            for path in class_paths:
                cls_name = path.rsplit(".", 1)[-1]
                if re.search(rf"\b{cls_name}\b", script_body_no_comments):
                    used_paths.add(path)
            class_paths = used_paths

            used_std = set()
            for mod in std_imports:
                if re.search(rf"\b{mod}\b", script_body_no_comments):
                    used_std.add(mod)
            std_imports = used_std

        # Parse class paths into import statements grouped by package
        sklearn_imports: dict[str, set[str]] = {}  # module → {classes}
        external_imports: dict[str, set[str]] = {}  # module → {classes}

        for path in class_paths:
            parts = path.rsplit(".", 1)
            if len(parts) != 2:
                continue
            module, cls = parts

            if module.startswith("sklearn") or module.startswith("imblearn"):
                sklearn_imports.setdefault(module, set()).add(cls)
            else:
                external_imports.setdefault(module, set()).add(cls)

        # Build import lines
        lines: list[str] = []

        # Group 0: stdlib imports extracted from code
        std_imports.discard('numpy')
        std_imports.discard('pandas')
        if std_imports:
            for mod in sorted(std_imports):
                lines.append(f"import {mod}")
            lines.append("")

        # Group 1: third-party (always needed)
        lines.append("import numpy as np")
        lines.append("import pandas as pd")
        lines.append("")

        # Group 2: sklearn (sorted by module)
        sklearn_always = [
            ("sklearn.model_selection", ["train_test_split"]),
        ]
        for mod, cls_list in sklearn_always:
            sklearn_imports.setdefault(mod, set()).update(cls_list)

        # Add RandomizedSearchCV if there are model candidates
        if candidates:
            sklearn_imports.setdefault(
                "sklearn.model_selection", set()
            ).add("RandomizedSearchCV")

        if sklearn_imports:
            for module in sorted(sklearn_imports):
                classes = sorted(sklearn_imports[module])
                lines.append(
                    f"from {module} import {', '.join(classes)}"
                )
            lines.append("")

        # Group 3: external ML libs
        if external_imports:
            for module in sorted(external_imports):
                classes = sorted(external_imports[module])
                lines.append(
                    f"from {module} import {', '.join(classes)}"
                )
            lines.append("")

        return "\n".join(lines)

    # ── Docstring ─────────────────────────────────────────────────

    @staticmethod
    def _build_docstring(
        profile: ProfileResult, gemini: GeminiResult,
    ) -> str:
        """Build the module docstring."""
        analyst = gemini.analyst_diagnostic
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        task = analyst.confirmed_task_type or "unknown"
        target = analyst.confirmed_target_column or "N/A"

        return textwrap.dedent(f'''\
            """
            Auto-Generated ML Pipeline — CogniPipe v{_VERSION}
            ===================================================
            Generated: {now}
            Task type: {task}
            Target column: {target}
            Rows: {profile.dataset.num_rows} | Columns: {profile.dataset.num_columns}

            This script was generated by CogniPipe. To use it:
            1. Replace the CSV path in the DATA LOADING section.
            2. Run: python pipeline.py
            """
        ''')

    # ── Configuration ─────────────────────────────────────────────

    @staticmethod
    def _build_config(gemini: GeminiResult) -> str:
        """Build the configuration constants block."""
        analyst = gemini.analyst_diagnostic
        target = analyst.confirmed_target_column or "target"
        task = analyst.confirmed_task_type or "unknown"

        return textwrap.dedent(f"""\
            # === CONFIGURATION ===
            TARGET_COLUMN = "{target}"
            TASK_TYPE = "{task}"
            RANDOM_STATE = 42
            TEST_SIZE = 0.2
            MODEL_CHOICE = 0  # 0=best, 1=second, 2=third
        """)

    # ── Data loading ──────────────────────────────────────────────

    @staticmethod
    def _build_loading_block(profile: ProfileResult) -> str:
        """Build the data loading block."""
        lines = [
            '# === DATA LOADING ===',
            '# Replace with your actual file path',
            'df = pd.read_csv("your_data.csv")',
            'print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")',
        ]

        lines.append('')
        return '\n'.join(lines) + '\n'

    # ── Feature engineering ───────────────────────────────────────

    @staticmethod
    def _build_feature_engineering_block(steps: list[FeatureStep]) -> str:
        """Build the feature engineering section.

        Each step gets a comment with its rationale and a try/except wrapper.
        """
        if not steps:
            return (
                "# === FEATURE ENGINEERING ===\n"
                "# No feature engineering steps prescribed.\n"
            )

        lines = ["# === FEATURE ENGINEERING ==="]
        for step in sorted(steps, key=lambda s: s.step_order):
            cols_str = ", ".join(f"'{c}'" for c in step.target_columns)
            lines.append(f"")
            lines.append(
                f"# Step {step.step_order}: {step.operation} on [{cols_str}]"
            )
            lines.append(f"# {step.rationale}")
            if step.sklearn_equivalent:
                lines.append(f"# sklearn: {step.sklearn_equivalent}")
            lines.append("try:")

            # Indent each line of the code snippet
            # Strip inline imports and fix E701 (inline returns)
            snippet = step.code_snippet.strip()
            snippet = re.sub(r"^( +)?if\s+(.*?):\s+(return\s+.*)$", r"\1if \2:\n\1    \3", snippet, flags=re.MULTILINE)
            has_executable = False
            for code_line in snippet.split("\n"):
                if re.match(r"^from\s+\S+\s+import\s+", code_line.lstrip()):
                    continue
                if re.match(r"^import\s+", code_line.lstrip()):
                    continue
                if code_line.strip() and not code_line.strip().startswith("#"):
                    has_executable = True
                lines.append(f"    {code_line}")
            
            if not has_executable:
                lines.append("    pass")

            # Error handling with column names
            lines.append("except Exception as e:")
            lines.append(
                f"    print(f\"Warning: {step.operation} on "
                f"[{cols_str}] failed: {{e}}\")"
            )
            lines.append(
                f"    print(\"Continuing with untransformed values.\")"
            )

        lines.append("")
        return "\n".join(lines) + "\n"

    # ── Train/test split ──────────────────────────────────────────

    @staticmethod
    def _build_split_block(ml: MLArchitectureRecommendation) -> str:
        """Build the train/test split block."""
        task = ml.task_type.value if isinstance(ml.task_type, TaskType) else str(ml.task_type)
        is_classification = "classification" in task.lower()

        lines = ["# === TRAIN/TEST SPLIT ==="]
        lines.append(
            "X = df.drop(columns=[TARGET_COLUMN], errors='ignore')"
        )
        lines.append("y = df[TARGET_COLUMN]")
        lines.append("")

        if is_classification:
            lines.append(
                "X_train, X_test, y_train, y_test = train_test_split("
            )
            lines.append(
                "    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,"
            )
            lines.append("    stratify=y,")
            lines.append(")")
        else:
            lines.append(
                "X_train, X_test, y_train, y_test = train_test_split("
            )
            lines.append(
                "    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,"
            )
            lines.append(")")

        lines.append("")
        return "\n".join(lines) + "\n"

    # ── Model block ───────────────────────────────────────────────

    @staticmethod
    def _build_model_block(candidates: list[ModelCandidate]) -> str:
        """Build the model training block with RandomizedSearchCV.

        Supports MODEL_CHOICE to select among candidates.
        """
        lines = ["# === MODEL ==="]
        lines.append("models = []")
        lines.append("")

        for i, cand in enumerate(sorted(candidates, key=lambda c: c.rank)):
            lines.append(f"# Candidate {cand.rank}: {cand.model_name}")
            lines.append(f"# {cand.rationale}")

            # Parse the class path for instantiation
            parts = cand.sklearn_class.rsplit(".", 1)
            cls_name = parts[-1] if len(parts) == 2 else cand.sklearn_class

            # Hyperparameter dict
            hp_str = json.dumps(cand.hyperparameters, indent=4)
            # Fix JSON literals to Python literals (safe because strings are double-quoted)
            hp_str = re.sub(r"\bnull\b", "None", hp_str)
            hp_str = re.sub(r"\btrue\b", "True", hp_str)
            hp_str = re.sub(r"\bfalse\b", "False", hp_str)
            lines.append(f"_hp_{i} = {hp_str}")
            lines.append(f"models.append((")
            lines.append(f"    '{cand.model_name}',")
            lines.append(f"    {cls_name}(),")
            lines.append(f"    _hp_{i},")
            lines.append(f"))")
            lines.append("")

        # Selection + RandomizedSearchCV
        lines.append("if not models:")
        lines.append("    raise ValueError(")
        lines.append("        \"No model candidates were generated. \"")
        lines.append("        \"Re-run CogniPipe to regenerate the pipeline.\"")
        lines.append("    )")
        lines.append(
            "selected_name, selected_model, selected_params = "
            "models[MODEL_CHOICE]"
        )
        lines.append(f"print(f\"Training: {{selected_name}}\")")
        lines.append("")
        lines.append("search = RandomizedSearchCV(")
        lines.append("    estimator=selected_model,")
        lines.append("    param_distributions=selected_params,")
        lines.append("    n_iter=20,")
        lines.append("    cv=5,")
        lines.append("    random_state=RANDOM_STATE,")
        lines.append("    n_jobs=-1,")
        lines.append("    verbose=1,")
        lines.append(")")
        lines.append("")
        lines.append("search.fit(X_train, y_train)")
        lines.append("best_model = search.best_estimator_")
        lines.append(
            "print(f\"Best params: {search.best_params_}\")"
        )
        lines.append("")

        return "\n".join(lines) + "\n"

    # ── Evaluation block ──────────────────────────────────────────

    @staticmethod
    def _build_evaluation_block(ml: MLArchitectureRecommendation) -> str:
        """Build evaluation metrics appropriate to task type."""
        task = ml.task_type.value if isinstance(ml.task_type, TaskType) else str(ml.task_type)
        is_classification = "classification" in task.lower()

        lines = [
            "# === EVALUATION ===",
            "if 'best_model' not in dir():",
            "    print('No model was trained — skipping evaluation.')",
            "else:",
        ]

        # All evaluation code is indented inside the else block
        eval_lines = [
            "y_pred = best_model.predict(X_test)",
            "",
        ]

        if is_classification:
            eval_lines.extend([
                "from sklearn.metrics import ("
                "accuracy_score, classification_report, "
                "confusion_matrix)",
                "",
                "accuracy = accuracy_score(y_test, y_pred)",
                "print(f\"\\nAccuracy: {accuracy:.4f}\")",
                "print(\"\\nClassification Report:\")",
                "print(classification_report(y_test, y_pred))",
                "print(\"\\nConfusion Matrix:\")",
                "print(confusion_matrix(y_test, y_pred))",
            ])
        else:
            eval_lines.extend([
                "from sklearn.metrics import ("
                "mean_absolute_error, mean_squared_error, r2_score)",
                "",
                "mae = mean_absolute_error(y_test, y_pred)",
                "rmse = mean_squared_error(y_test, y_pred) ** 0.5",
                "r2 = r2_score(y_test, y_pred)",
                "print(f\"\\nMAE:  {mae:.4f}\")",
                "print(f\"RMSE: {rmse:.4f}\")",
                "print(f\"R²:   {r2:.4f}\")",
            ])

        # Indent eval_lines inside the else block
        for line in eval_lines:
            lines.append(f"    {line}" if line else "")

        lines.append("")
        return "\n".join(lines) + "\n"

    # ── Main block ────────────────────────────────────────────────

    @staticmethod
    def _build_main_block() -> str:
        """Build the if __name__ == '__main__' block."""
        return textwrap.dedent("""\
            # === MAIN ===
            # This script is designed to be run directly:
            #   python pipeline.py
            #
            # Before running, replace 'your_data.csv' with
            # the path to your actual dataset.
        """)

    # ── Script assembly ───────────────────────────────────────────

    @staticmethod
    def _assemble_script(*blocks: str) -> str:
        """Join all script blocks with consistent spacing."""
        return "\n\n".join(block.rstrip() for block in blocks if block.strip()) + "\n"

    # ── Notebook generation ───────────────────────────────────────

    @staticmethod
    def _build_notebook(script: str, gemini: GeminiResult) -> str:
        """Build a Jupyter Notebook mirroring the script structure.

        Each ``# === SECTION ===`` header becomes:
        1. A markdown cell explaining the section.
        2. A code cell with the section's code.

        The diagnostic summary goes in a markdown cell at the top.
        """
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }

        # Title + diagnostic summary
        analyst = gemini.analyst_diagnostic
        title_md = (
            "# 🧠 CogniPipe — Auto-Generated ML Pipeline\n\n"
            f"**Task:** {analyst.confirmed_task_type}  \n"
            f"**Target:** {analyst.confirmed_target_column or 'N/A'}\n\n"
            "---\n\n"
            "## Dataset Diagnostic Summary\n\n"
            f"{analyst.diagnostic_summary}\n"
        )
        nb.cells.append(nbformat.v4.new_markdown_cell(title_md))

        # Section descriptions for markdown cells
        section_descriptions: dict[str, str] = {
            "CONFIGURATION": (
                "## Configuration\n\n"
                "Set the target column, random seed, test split ratio, "
                "and which model candidate to use."
            ),
            "DATA LOADING": (
                "## Data Loading\n\n"
                "Load your dataset. **Replace `your_data.csv`** with "
                "the path to your actual CSV file."
            ),
            "FEATURE ENGINEERING": (
                "## Feature Engineering\n\n"
                "Apply AI-prescribed transformations to improve "
                "model performance. Each step includes a rationale "
                "based on the statistical profile of the data."
            ),
            "PREPROCESSING": (
                "## Preprocessing\n\n"
                "Standard preprocessing steps: imputation, scaling, "
                "and encoding applied before model training."
            ),
            "TRAIN/TEST SPLIT": (
                "## Train/Test Split\n\n"
                "Split the data into training and test sets for "
                "model evaluation."
            ),
            "MODEL": (
                "## Model Training\n\n"
                "Train the selected model using RandomizedSearchCV "
                "for hyperparameter tuning."
            ),
            "EVALUATION": (
                "## Evaluation\n\n"
                "Evaluate the trained model on the test set using "
                "task-appropriate metrics."
            ),
            "MAIN": (
                "## Run Info\n\n"
                "This notebook mirrors the auto-generated pipeline "
                "script."
            ),
        }

        # Split script into sections by "# === SECTION ==="
        # Skip the docstring (first block before any section header)
        section_pattern = re.compile(r"^# === (.+?) ===$", re.MULTILINE)
        parts = section_pattern.split(script)

        # parts[0] = docstring + imports (before first header)
        # parts[1] = section name, parts[2] = section code, etc.

        # Add imports as a code cell (everything before first section)
        preamble = parts[0].strip()
        if preamble:
            nb.cells.append(
                nbformat.v4.new_markdown_cell(
                    "## Setup & Imports\n\n"
                    "Import all required libraries."
                )
            )
            # Strip the docstring from the preamble for the notebook
            # (it's redundant since we have the markdown title cell)
            preamble_lines = preamble.split("\n")
            in_docstring = False
            clean_lines = []
            for line in preamble_lines:
                if '"""' in line and not in_docstring:
                    in_docstring = True
                    if line.count('"""') >= 2:
                        in_docstring = False
                    continue
                elif '"""' in line and in_docstring:
                    in_docstring = False
                    continue
                elif in_docstring:
                    continue
                else:
                    clean_lines.append(line)

            clean_preamble = "\n".join(clean_lines).strip()
            if clean_preamble:
                nb.cells.append(
                    nbformat.v4.new_code_cell(clean_preamble)
                )

        # Process paired (section_name, section_code) entries
        for i in range(1, len(parts), 2):
            section_name = parts[i].strip()
            section_code = parts[i + 1].strip() if i + 1 < len(parts) else ""

            # Markdown description
            md = section_descriptions.get(
                section_name,
                f"## {section_name.title()}\n",
            )
            nb.cells.append(nbformat.v4.new_markdown_cell(md))

            # Code cell
            if section_code:
                nb.cells.append(nbformat.v4.new_code_cell(section_code))

        return json.dumps(nbformat.from_dict(nb), indent=1)

    # ── Requirements.txt ──────────────────────────────────────────

    @staticmethod
    def _build_requirements(
        steps: list[FeatureStep],
        candidates: list[ModelCandidate],
    ) -> str:
        """Build a requirements.txt from actual imports used.

        Rules:
        - Always include pandas, numpy, scikit-learn.
        - Only include keras/tensorflow if a neural network model
          is among the candidates.
        - Pin minor versions for reproducibility.
        """
        # Core packages (always needed)
        packages: OrderedDict[str, str] = OrderedDict()
        packages["pandas"] = _VERSION_PINS.get("pandas", "pandas")
        packages["numpy"] = _VERSION_PINS.get("numpy", "numpy")
        packages["scikit-learn"] = _VERSION_PINS.get(
            "scikit-learn", "scikit-learn"
        )
        packages["matplotlib"] = _VERSION_PINS.get(
            "matplotlib", "matplotlib"
        )

        # Scan FeatureStep sklearn_equivalent fields
        all_paths: set[str] = set()
        for step in steps:
            if step.sklearn_equivalent:
                all_paths.add(step.sklearn_equivalent)
        for cand in candidates:
            if cand.sklearn_class:
                all_paths.add(cand.sklearn_class)

        # Derive pip packages from top-level module names
        for path in all_paths:
            top_module = path.split(".")[0]
            pip_name = _PACKAGE_MAP.get(top_module)
            if pip_name and pip_name not in packages:
                packages[pip_name] = _VERSION_PINS.get(pip_name, pip_name)

        # Build requirements text with grouped comments
        lines: list[str] = []
        lines.append("# === Core ===")
        for pkg in ["pandas", "numpy", "scikit-learn", "matplotlib"]:
            if pkg in packages:
                lines.append(packages[pkg])

        # Optional ML packages
        optional = [
            k for k in packages
            if k not in {"pandas", "numpy", "scikit-learn", "matplotlib"}
        ]
        if optional:
            lines.append("")
            lines.append("# === ML Libraries ===")
            for pkg in optional:
                lines.append(packages[pkg])

        lines.append("")
        return "\n".join(lines)
