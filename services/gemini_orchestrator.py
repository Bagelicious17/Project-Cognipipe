"""
CogniPipe — Layer 2: GeminiOrchestrator
=========================================

Three-chain orchestration that converts a ``ProfileResult`` into a
``GeminiResult`` using the Gemini API.

Chain 1 (Analyst)   → AnalystDiagnostic
Chain 2 (Engineer)  → FeatureEngineeringPrescription
Chain 3 (Architect) → MLArchitectureRecommendation

Rules:
- No pandas in this file.  Ever.
- Prompts live in ``gemini_prompts.py``.
- Every Gemini call is retried up to 3 times on JSON parse failure.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from models.schemas import (
    AnalystDiagnostic,
    ChainTokenUsage,
    FeatureEngineeringPrescription,
    GeminiResult,
    MLArchitectureRecommendation,
    ProfileResult,
)
from services.gemini_prompts import (
    CHAIN_1_SYSTEM,
    CHAIN_1_USER,
    CHAIN_2_SYSTEM,
    CHAIN_2_USER,
    CHAIN_3_SYSTEM,
    CHAIN_3_USER,
    RETRY_PROMPT,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Custom exception
# ──────────────────────────────────────────────────────────────────────

class GeminiOrchestrationError(Exception):
    """Raised when a Gemini chain call fails after all retries.

    Attributes:
        chain_number: Which chain failed (1, 2, or 3).
        original_error: The underlying exception.
    """

    def __init__(self, chain_number: int, message: str, original_error: Exception | None = None):
        self.chain_number = chain_number
        self.original_error = original_error
        super().__init__(f"Chain {chain_number} failed: {message}")


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

class GeminiOrchestrator:
    """Runs the 3-chain Gemini orchestration pipeline.

    Usage::

        from config import settings
        orch = GeminiOrchestrator(api_key=settings.gemini_api_key)
        result: GeminiResult = orch.run(profile_result)
    """

    MAX_RETRIES = 3

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """Initialise the orchestrator with a Gemini API key.

        Args:
            api_key: Google Gemini API key.
            model: Model name (default: ``gemini-2.5-flash``).
        """
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._token_usage: list[ChainTokenUsage] = []

    # ── public entry point ────────────────────────────────────────

    def run(self, profile: ProfileResult) -> GeminiResult:
        """Execute all three chains sequentially and return a ``GeminiResult``.

        Args:
            profile: The ``ProfileResult`` from Layer 1.

        Returns:
            Fully populated ``GeminiResult``.

        Raises:
            GeminiOrchestrationError: If any chain fails after retries.
        """
        t0 = time.perf_counter()
        self._token_usage = []
        raw_responses: dict[str, Any] = {}

        # Chain 1 — Analyst
        chain1_dict = self._run_chain_1(profile)
        raw_responses["chain_1_analyst"] = chain1_dict
        analyst = AnalystDiagnostic.model_validate(chain1_dict)

        # Chain 2 — Feature Engineer
        chain2_prescription = self._run_chain_2(profile, chain1_dict)
        raw_responses["chain_2_feature_engineer"] = chain2_prescription.model_dump()

        # Chain 3 — ML Architect
        chain3_architecture = self._run_chain_3(
            profile, chain1_dict, chain2_prescription
        )
        raw_responses["chain_3_ml_architect"] = chain3_architecture.model_dump()

        duration = round(time.perf_counter() - t0, 4)

        return GeminiResult(
            analyst_diagnostic=analyst,
            feature_engineering=chain2_prescription,
            ml_architecture=chain3_architecture,
            raw_responses=raw_responses,
            token_usage=self._token_usage,
            orchestration_duration_seconds=duration,
        )

    # ── Chain 1: Analyst ──────────────────────────────────────────

    def _run_chain_1(self, profile: ProfileResult) -> dict:
        """Run the analyst chain to produce a dataset diagnostic.

        Args:
            profile: Full profiling result from Layer 1.

        Returns:
            Raw parsed JSON dict matching ``AnalystDiagnostic`` shape.
        """
        context = self._profile_to_prompt_context(profile)
        user_prompt = CHAIN_1_USER.format(profile_context=context)

        return self._call_gemini(
            chain_number=1,
            chain_name="chain_1_analyst",
            system_prompt=CHAIN_1_SYSTEM,
            user_prompt=user_prompt,
        )

    # ── Chain 2: Feature Engineer ─────────────────────────────────

    def _run_chain_2(
        self, profile: ProfileResult, chain1: dict
    ) -> FeatureEngineeringPrescription:
        """Run the feature engineering chain.

        Args:
            profile: Full profiling result.
            chain1: Raw Chain 1 analyst output dict.

        Returns:
            Validated ``FeatureEngineeringPrescription``.
        """
        context = self._profile_to_prompt_context(profile)

        critical_issues = "\n".join(
            f"- {issue}" for issue in chain1.get("critical_issues", [])
        ) or "- None identified"

        priority_cols = ", ".join(
            chain1.get("priority_columns_for_engineering", [])
        ) or "None specified"

        column_roles = json.dumps(
            chain1.get("column_roles", {}), indent=2
        )

        target_col = chain1.get("confirmed_target_column")
        target_context = (
            f"'{target_col}'"
            if target_col
            else "not identified (unsupervised task or ambiguous dataset)"
        )

        user_prompt = CHAIN_2_USER.format(
            confirmed_task_type=chain1.get("confirmed_task_type") or "unknown",
            confirmed_target_column=target_context,
            critical_issues=critical_issues,
            priority_columns=priority_cols,
            column_roles=column_roles,
            profile_context=context,
        )

        raw = self._call_gemini(
            chain_number=2,
            chain_name="chain_2_feature_engineer",
            system_prompt=CHAIN_2_SYSTEM,
            user_prompt=user_prompt,
        )

        prescription = FeatureEngineeringPrescription.model_validate(raw)

        # Filter out feature steps on columns that were identified as drop_candidate or id
        column_roles_dict = chain1.get("column_roles", {})
        valid_steps = []
        for step in prescription.steps:
            # Check if any target column in the step is a drop candidate
            is_valid = True
            for col in step.target_columns:
                role = column_roles_dict.get(col, "")
                if role in ["drop_candidate", "id"]:
                    is_valid = False
                    break
            if is_valid:
                valid_steps.append(step)

        prescription.steps = valid_steps
        return prescription

    # ── Chain 3: ML Architect ─────────────────────────────────────

    def _run_chain_3(
        self,
        profile: ProfileResult,
        chain1: dict,
        chain2: FeatureEngineeringPrescription,
    ) -> MLArchitectureRecommendation:
        """Run the ML architecture chain.

        Args:
            profile: Full profiling result.
            chain1: Raw Chain 1 analyst output dict.
            chain2: Validated feature engineering prescription.

        Returns:
            Validated ``MLArchitectureRecommendation``.
        """
        context = self._profile_to_prompt_context(profile)

        critical_issues = "\n".join(
            f"- {issue}" for issue in chain1.get("critical_issues", [])
        ) or "- None identified"

        fe_summary = chain2.summary or "No feature engineering summary available."
        fe_steps = ", ".join(
            f"{s.operation} on {s.target_columns}" for s in chain2.steps
        ) or "No steps defined."
        feature_eng_text = f"{fe_summary}\nSteps: {fe_steps}"

        target_col = chain1.get("confirmed_target_column")
        target_context = (
            f"'{target_col}'"
            if target_col
            else (
                "not identified. Recommend both supervised approaches "
                "(assuming the last numerical column as target) and "
                "unsupervised approaches (clustering/anomaly detection) "
                "with a clear note explaining the ambiguity"
            )
        )

        user_prompt = CHAIN_3_USER.format(
            confirmed_task_type=chain1.get("confirmed_task_type") or "unknown",
            confirmed_target_column=target_context,
            critical_issues=critical_issues,
            feature_engineering_summary=feature_eng_text,
            profile_context=context,
        )

        raw = self._call_gemini(
            chain_number=3,
            chain_name="chain_3_ml_architect",
            system_prompt=CHAIN_3_SYSTEM,
            user_prompt=user_prompt,
        )

        return MLArchitectureRecommendation.model_validate(raw)

    # ── Core Gemini caller with retry ─────────────────────────────

    def _call_gemini(
        self,
        chain_number: int,
        chain_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """Call the Gemini API with retry logic for JSON parse failures.

        Args:
            chain_number: Chain identifier (1, 2, or 3).
            chain_name: Human-readable chain name for logging.
            system_prompt: System instruction with persona.
            user_prompt: User prompt with task and context.

        Returns:
            Parsed JSON dict from the Gemini response.

        Raises:
            GeminiOrchestrationError: If all retries are exhausted or
                the API returns an error.
        """
        current_prompt = user_prompt
        last_error: Exception | None = None
        raw_text = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(
                    "Chain %d (%s) — attempt %d/%d",
                    chain_number, chain_name, attempt, self.MAX_RETRIES,
                )

                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=current_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                    ),
                )

                # Track token usage
                self._track_tokens(chain_name, response)

                # Extract text
                raw_text = response.text.strip()

                # Strip markdown fences if Gemini ignores our instruction
                raw_text = self._strip_markdown_fences(raw_text)

                # Parse JSON
                parsed = json.loads(raw_text)
                logger.info("Chain %d — JSON parsed successfully.", chain_number)
                return parsed

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "Chain %d — JSON parse failed (attempt %d): %s",
                    chain_number, attempt, str(e),
                )
                # Build retry prompt with the raw response for self-correction
                current_prompt = RETRY_PROMPT.format(
                    error=str(e),
                    raw_response=raw_text[:3000],
                )

            except Exception as e:
                logger.error(
                    "Chain %d — API error: %s", chain_number, str(e)
                )
                raise GeminiOrchestrationError(
                    chain_number=chain_number,
                    message=str(e),
                    original_error=e,
                ) from e

        # All retries exhausted
        raise GeminiOrchestrationError(
            chain_number=chain_number,
            message=f"Failed to get valid JSON after {self.MAX_RETRIES} attempts. Last error: {last_error}",
            original_error=last_error,
        )

    # ── Token tracking ────────────────────────────────────────────

    def _track_tokens(self, chain_name: str, response: Any) -> None:
        """Extract and log token usage from a Gemini API response.

        Args:
            chain_name: Identifier for the chain.
            response: Raw Gemini response object.
        """
        try:
            usage = response.usage_metadata
            input_tok = getattr(usage, "prompt_token_count", 0) or 0
            output_tok = getattr(usage, "candidates_token_count", 0) or 0
            total_tok = input_tok + output_tok

            self._token_usage.append(ChainTokenUsage(
                chain_name=chain_name,
                input_tokens=input_tok,
                output_tokens=output_tok,
                total_tokens=total_tok,
            ))

            logger.info(
                "%s tokens — in: %d, out: %d, total: %d",
                chain_name, input_tok, output_tok, total_tok,
            )
        except Exception:
            logger.warning("Could not extract token usage for %s", chain_name)
            self._token_usage.append(ChainTokenUsage(chain_name=chain_name))

    # ── Profile → prompt context ──────────────────────────────────

    MAX_COLUMNS_IN_CONTEXT = 30  # truncation threshold for wide datasets

    @staticmethod
    def _profile_to_prompt_context(
        profile: ProfileResult,
        *,
        max_columns: int = 30,
    ) -> str:
        """Convert a ``ProfileResult`` into a condensed JSON string for prompts.

        Rules:
        - Excludes columns whose inferred semantic type is 'id'.
        - Rounds all floats to 4 decimal places.
        - Includes sample_values for every column.
        - Includes dataset-level flags.
        - Targets ~3000 tokens (roughly 12KB of text).
        - If more than ``max_columns`` non-ID columns exist, keeps
          target + highest-missing + random sample to fit the budget.

        Args:
            profile: The full ``ProfileResult``.
            max_columns: Maximum number of columns to include.

        Returns:
            JSON string ready for injection into a prompt.
        """
        # Collect non-ID columns
        non_id_cols = [
            col for col in profile.columns
            if col.inferred_semantic_type.value != "id"
        ]

        # Truncation for wide datasets
        truncated = False
        if len(non_id_cols) > max_columns:
            truncated = True
            target_name = profile.dataset.suspected_target_column
            # Always keep the target column
            target_cols = [c for c in non_id_cols if c.column_name == target_name]
            other_cols = [c for c in non_id_cols if c.column_name != target_name]
            # Sort by missing % descending (most problematic first)
            other_cols.sort(
                key=lambda c: c.missing.missing_percentage, reverse=True
            )
            non_id_cols = target_cols + other_cols[: max_columns - len(target_cols)]

        columns_data = []
        for col in non_id_cols:
            col_info: dict[str, Any] = {
                "name": col.column_name,
                "dtype": col.dtype,
                "semantic_type": col.inferred_semantic_type.value,
                "missing_pct": round(col.missing.missing_percentage, 4),
                "missing_pattern": col.missing.missing_pattern.value,
                "nunique": col.nunique,
                "sample_values": col.sample_values[:5],
            }

            if col.numerical_stats:
                ns = col.numerical_stats
                col_info["stats"] = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in {
                        "mean": ns.mean,
                        "median": ns.median,
                        "std": ns.std,
                        "min": ns.min,
                        "max": ns.max,
                        "skewness": ns.skewness,
                        "kurtosis": ns.kurtosis,
                        "q1": ns.q1,
                        "q3": ns.q3,
                        "iqr": ns.iqr,
                        "outlier_iqr": ns.outlier_count_iqr,
                        "outlier_zscore": ns.outlier_count_zscore,
                        "zero_count": ns.zero_count,
                        "negative_count": ns.negative_count,
                        "is_log_distributed": ns.is_likely_log_distributed,
                    }.items()
                    if v is not None
                }

            if col.categorical_stats:
                cs = col.categorical_stats
                col_info["stats"] = {
                    "cardinality": cs.cardinality,
                    "cardinality_ratio": round(cs.cardinality_ratio, 4),
                    "top_values": dict(list(cs.top_10_values.items())[:5]),
                    "rare_categories": cs.rare_category_count,
                }

            if col.datetime_stats:
                ds_col = col.datetime_stats
                col_info["stats"] = {
                    k: v for k, v in {
                        "min_date": ds_col.min_date,
                        "max_date": ds_col.max_date,
                        "span_days": ds_col.time_span_days,
                        "frequency": ds_col.inferred_frequency,
                    }.items()
                    if v is not None
                }

            columns_data.append(col_info)

        ds = profile.dataset
        dataset_info: dict[str, Any] = {
            "rows": ds.num_rows,
            "columns": ds.num_columns,
            "is_time_series": ds.is_time_series,
            "suspected_target": ds.suspected_target_column,
            "likely_task_type": ds.likely_task_type.value,
            "class_imbalance_ratio": ds.class_imbalance_ratio,
            "high_correlation_pairs": [
                {
                    "a": p.column_a,
                    "b": p.column_b,
                    "pearson": round(p.pearson, 4) if p.pearson else None,
                }
                for p in ds.high_correlation_pairs[:10]
            ],
            "leakage_risks": [
                {"col": r.column_name, "reason": r.reason.value}
                for r in ds.data_leakage_risks
            ],
            "duplicate_rows": ds.duplicate_row_count,
        }
        if truncated:
            dataset_info["note"] = (
                f"Context truncated: showing {len(columns_data)} of "
                f"{ds.num_columns} columns (target + highest-missing)."
            )

        context = {
            "dataset": dataset_info,
            "columns": columns_data,
        }

        return json.dumps(context, indent=2, default=str)

    # ── Utility ───────────────────────────────────────────────────

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences if present.

        Gemini sometimes wraps JSON in ```json ... ``` despite
        instructions not to.

        Args:
            text: Raw response text.

        Returns:
            Cleaned text without fences.
        """
        if text.startswith("```"):
            first_newline = text.index("\n") if "\n" in text else len(text)
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
