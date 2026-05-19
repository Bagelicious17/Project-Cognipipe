"""
CogniPipe — Layer 1: DataProfiler
===================================
Produces a ``ProfileResult`` from a raw ``pandas.DataFrame``.
Uses ONLY pandas, numpy, scipy.  No AI calls.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from models.schemas import (
    CategoricalStats,
    ColumnProfile,
    CorrelationMethod,
    CorrelationPair,
    DataLeakageRisk,
    DatasetFlags,
    DatetimeStats,
    LeakageReason,
    MissingInfo,
    MissingPattern,
    NumericalStats,
    ProfileResult,
    SemanticType,
    TaskType,
)

logger = logging.getLogger(__name__)

# ── Regex patterns for semantic-type heuristics ──────────────────────

_ID_PAT = re.compile(r"(^id$|_id$|_key$|_index$|^index$|^key$)", re.I)
_DT_PAT = re.compile(r"(date|time|timestamp|year|month|day_of)", re.I)
_CYCLIC_PAT = re.compile(r"(hour|day|month|quarter|week|season)", re.I)
_TARGET_PAT = re.compile(
    r"(target|label|class|outcome|churn|price|sales|revenue|survived|y_)", re.I
)
_POSTHOC_PAT = re.compile(
    r"(predicted|prediction|forecast|_after|_result|_output)", re.I
)


class DataProfiler:
    """Stateless profiler that converts a DataFrame into a ``ProfileResult``.

    Usage::

        profiler = DataProfiler()
        result: ProfileResult = profiler.profile(df)
        json_blob = result.model_dump()
    """

    VERSION = "0.1.0"

    # ── public entry point ────────────────────────────────────────

    def profile(self, df: pd.DataFrame) -> ProfileResult:
        """Profile the entire DataFrame and return a ``ProfileResult``.

        Args:
            df: Raw pandas DataFrame to analyse.

        Returns:
            Fully populated ``ProfileResult`` ready for JSON serialization.
        """
        t0 = time.perf_counter()
        n_rows, n_cols = df.shape

        # Per-column profiling (isolated per column)
        col_profiles: list[ColumnProfile] = []
        for col in df.columns:
            try:
                cp = self._profile_column(df, col, n_rows)
                col_profiles.append(cp)
            except Exception:
                logger.exception("Failed to profile column '%s', using fallback", col)
                col_profiles.append(self._fallback_column(df, col, n_rows))

        # Dataset-level flags
        dataset = self._build_dataset_flags(df, col_profiles, n_rows, n_cols)

        return ProfileResult(
            columns=col_profiles,
            dataset=dataset,
            profiling_duration_seconds=round(time.perf_counter() - t0, 4),
            profiler_version=self.VERSION,
        )

    # ── column-level profiling ────────────────────────────────────

    def _profile_column(
        self, df: pd.DataFrame, col: str, n_rows: int
    ) -> ColumnProfile:
        """Build a full ``ColumnProfile`` for a single column.

        Args:
            df: The full DataFrame (needed for MAR correlation checks).
            col: Column name to profile.
            n_rows: Total row count.
        """
        series = df[col]
        dtype_str = str(series.dtype)
        nunique = int(series.nunique())

        # Missing info
        missing = self._compute_missing(df, col, n_rows)

        # Semantic type
        sem_type = self._infer_semantic_type(series, col, nunique, n_rows)

        # Try datetime conversion if hinted
        if sem_type == SemanticType.DATETIME:
            try:
                series = pd.to_datetime(series, errors="coerce")
            except Exception:
                sem_type = SemanticType.NOMINAL

        # Type-specific stats
        num_stats = None
        cat_stats = None
        dt_stats = None

        if sem_type in (SemanticType.CONTINUOUS, SemanticType.CYCLICAL):
            num_stats = self._compute_numerical(series)
        elif sem_type == SemanticType.ORDINAL:
            num_stats = self._compute_numerical(pd.to_numeric(series, errors="coerce"))
            cat_stats = self._compute_categorical(series, n_rows, nunique)
        elif sem_type in (SemanticType.NOMINAL, SemanticType.ID,
                          SemanticType.TARGET_CANDIDATE, SemanticType.UNKNOWN):
            # Numeric columns typed as target/ID still get num stats
            if pd.api.types.is_numeric_dtype(series):
                num_stats = self._compute_numerical(series)
            else:
                cat_stats = self._compute_categorical(series, n_rows, nunique)
        elif sem_type == SemanticType.DATETIME:
            dt_stats = self._compute_datetime(series)

        sample = self._sample_values(series)

        return ColumnProfile(
            column_name=col,
            dtype=dtype_str,
            inferred_semantic_type=sem_type,
            missing=missing,
            numerical_stats=num_stats,
            categorical_stats=cat_stats,
            datetime_stats=dt_stats,
            nunique=nunique,
            sample_values=sample,
        )

    def _fallback_column(
        self, df: pd.DataFrame, col: str, n_rows: int
    ) -> ColumnProfile:
        """Minimal fallback profile when full profiling fails for a column."""
        series = df[col]
        mc = int(series.isna().sum())
        return ColumnProfile(
            column_name=col,
            dtype=str(series.dtype),
            inferred_semantic_type=SemanticType.UNKNOWN,
            missing=MissingInfo(
                missing_count=mc,
                missing_percentage=round(mc / max(n_rows, 1) * 100, 4),
                missing_pattern=MissingPattern.NONE if mc == 0 else MissingPattern.MCAR,
            ),
            nunique=int(series.nunique()),
            sample_values=self._sample_values(series),
        )

    # ── semantic type inference ───────────────────────────────────

    def _infer_semantic_type(
        self, series: pd.Series, col: str, nunique: int, n_rows: int
    ) -> SemanticType:
        """Apply heuristic rules to classify a column's semantic role.

        Rules are applied in priority order so higher-confidence checks
        win over more general ones.
        """
        card_ratio = nunique / max(n_rows, 1)

        # 1. Datetime check (before ID so 'signup_date' isn't caught by cardinality)
        if pd.api.types.is_datetime64_any_dtype(series):
            return SemanticType.DATETIME
        if _DT_PAT.search(col):
            return SemanticType.DATETIME

        # 2. ID check (name-based first, then cardinality fallback)
        if _ID_PAT.search(col):
            return SemanticType.ID

        # 3. Target candidate check (before cardinality-based ID)
        if _TARGET_PAT.search(col):
            return SemanticType.TARGET_CANDIDATE

        # 4. High-cardinality ID fallback (only if not datetime/target)
        if card_ratio > 0.95:
            return SemanticType.ID

        # 4. Cyclical check
        if _CYCLIC_PAT.search(col):
            return SemanticType.CYCLICAL
        if (pd.api.types.is_integer_dtype(series)
                and nunique <= 31
                and series.dropna().min() >= 0
                and nunique >= 2):
            # Could be cyclical but only if name hints at it
            if _CYCLIC_PAT.search(col):
                return SemanticType.CYCLICAL

        # 5. Ordinal: few unique integers or object with numeric-like values
        if 2 <= nunique <= 15:
            if pd.api.types.is_integer_dtype(series):
                return SemanticType.ORDINAL
            if series.dtype == object:
                # Check if values look numeric
                non_null = series.dropna()
                if len(non_null) > 0:
                    numeric_frac = pd.to_numeric(non_null, errors="coerce").notna().mean()
                    if numeric_frac > 0.8:
                        return SemanticType.ORDINAL

        # 6. Nominal: object dtype catch-all
        if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
            return SemanticType.NOMINAL

        # 7. Continuous: float or high-cardinality int
        if pd.api.types.is_float_dtype(series):
            return SemanticType.CONTINUOUS
        if pd.api.types.is_integer_dtype(series) and card_ratio > 0.05:
            return SemanticType.CONTINUOUS

        # 8. Bool
        if pd.api.types.is_bool_dtype(series):
            return SemanticType.ORDINAL

        return SemanticType.UNKNOWN

    # ── missing value analysis ────────────────────────────────────

    def _compute_missing(
        self, df: pd.DataFrame, col: str, n_rows: int
    ) -> MissingInfo:
        """Compute missing-value count, percentage, and pattern heuristic.

        Pattern rules:
        - none: 0 % missing
        - MCAR: < 5 % AND missingness uncorrelated with other columns
        - MAR:  missingness indicator correlates (|r| > 0.3) with any
                other numeric column
        - MNAR: missing values concentrate in extreme quantiles
        """
        series = df[col]
        mc = int(series.isna().sum())
        pct = round(mc / max(n_rows, 1) * 100, 4)

        if mc == 0:
            return MissingInfo(
                missing_count=0,
                missing_percentage=0.0,
                missing_pattern=MissingPattern.NONE,
            )

        pattern = MissingPattern.MCAR  # default for non-zero

        # Check MAR: correlate binary missingness indicator with other numeric cols.
        # If |r| > 0.3 for any pair, missingness depends on another variable → MAR.
        try:
            indicator = series.isna().astype(int)
            num_cols = df.select_dtypes(include="number").columns.tolist()
            for c in num_cols:
                if c == col:
                    continue
                if df[c].isna().sum() == len(df):
                    continue  # skip all-null columns
                corr = indicator.corr(df[c])
                if corr is not None and not np.isnan(corr) and abs(corr) > 0.3:
                    pattern = MissingPattern.MAR
                    break
        except Exception:
            pass

        # Check MNAR: missing values concentrate in extreme quantiles.
        # Strategy: use a correlated numeric column as a proxy to see if
        # the rows where this column is missing tend to have extreme values
        # in the proxy.  Fallback: check if the column's own non-null
        # distribution is heavily skewed (suggesting tail values are missing).
        if pattern != MissingPattern.MAR and pd.api.types.is_numeric_dtype(series):
            try:
                non_null = series.dropna()
                null_mask = series.isna()
                if len(non_null) > 10 and null_mask.sum() > 0:
                    # Check proxy columns: for each numeric col, see if the
                    # values at the missing-row indices are in the top/bottom 10%
                    mnar_detected = False
                    for c in df.select_dtypes(include="number").columns:
                        if c == col or df[c].isna().sum() > len(df) * 0.5:
                            continue
                        proxy = df[c]
                        q10 = proxy.quantile(0.10)
                        q90 = proxy.quantile(0.90)
                        proxy_at_missing = proxy[null_mask].dropna()
                        if len(proxy_at_missing) < 3:
                            continue
                        extreme_frac = (
                            (proxy_at_missing <= q10) | (proxy_at_missing >= q90)
                        ).mean()
                        if extreme_frac > 0.5:  # >50% of missing rows are extreme
                            mnar_detected = True
                            break
                    # Fallback: heavy right-skew with all positive values
                    if not mnar_detected and non_null.skew() > 2.0 and pct > 3.0:
                        mnar_detected = True
                    if mnar_detected:
                        pattern = MissingPattern.MNAR
            except Exception:
                pass

        # Final: if < 5% and still MCAR, confirm it
        if pattern == MissingPattern.MCAR and pct >= 5.0:
            # Over 5% missing but no MAR/MNAR detected — still call it MCAR
            # but this is a weaker classification
            pass

        return MissingInfo(
            missing_count=mc,
            missing_percentage=pct,
            missing_pattern=pattern,
        )

    # ── numerical statistics ──────────────────────────────────────

    def _compute_numerical(self, series: pd.Series) -> NumericalStats:
        """Compute descriptive stats for a numeric series.

        Handles edge cases: all-null, single-value, zero-variance.
        """
        clean = pd.to_numeric(series, errors="coerce").dropna()

        if len(clean) == 0:
            return NumericalStats()

        desc = clean.describe()
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1

        # Outliers — IQR method
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_iqr = int(((clean < lower) | (clean > upper)).sum())

        # Outliers — z-score method
        std_val = float(clean.std())
        if std_val > 0:
            z = np.abs((clean - clean.mean()) / std_val)
            outlier_z = int((z > 3).sum())
        else:
            outlier_z = 0

        skew = float(sp_stats.skew(clean, nan_policy="omit"))
        kurt = float(sp_stats.kurtosis(clean, nan_policy="omit"))

        all_positive = bool((clean > 0).all())
        is_log = bool(skew > 1.5 and all_positive)

        return NumericalStats(
            mean=round(float(clean.mean()), 6),
            median=round(float(clean.median()), 6),
            std=round(std_val, 6),
            min=round(float(clean.min()), 6),
            max=round(float(clean.max()), 6),
            skewness=round(skew, 6),
            kurtosis=round(kurt, 6),
            q1=round(q1, 6),
            q3=round(q3, 6),
            iqr=round(iqr, 6),
            outlier_count_iqr=outlier_iqr,
            outlier_count_zscore=outlier_z,
            zero_count=int((clean == 0).sum()),
            negative_count=int((clean < 0).sum()),
            is_likely_log_distributed=is_log,
        )

    # ── categorical statistics ────────────────────────────────────

    def _compute_categorical(
        self, series: pd.Series, n_rows: int, nunique: int
    ) -> CategoricalStats:
        """Compute descriptive stats for a categorical series."""
        card_ratio = nunique / max(n_rows, 1)

        vc = series.value_counts(dropna=True)
        top_10 = {str(k): int(v) for k, v in vc.head(10).items()}

        # Rare categories: frequency < 1% of total rows
        threshold = n_rows * 0.01
        rare = int((vc < threshold).sum()) if len(vc) > 0 else 0

        return CategoricalStats(
            cardinality=nunique,
            cardinality_ratio=round(card_ratio, 6),
            top_10_values=top_10,
            rare_category_count=rare,
        )

    # ── datetime statistics ───────────────────────────────────────

    def _compute_datetime(self, series: pd.Series) -> DatetimeStats:
        """Compute descriptive stats for a datetime series."""
        clean = series.dropna()

        if len(clean) == 0:
            return DatetimeStats()

        mn = clean.min()
        mx = clean.max()
        span = (mx - mn).total_seconds() / 86400.0

        # Infer frequency
        freq = self._infer_dt_frequency(clean)

        return DatetimeStats(
            min_date=str(mn.isoformat()) if pd.notna(mn) else None,
            max_date=str(mx.isoformat()) if pd.notna(mx) else None,
            time_span_days=round(span, 2),
            inferred_frequency=freq,
        )

    @staticmethod
    def _infer_dt_frequency(dt_series: pd.Series) -> str:
        """Guess time-series frequency from a sorted datetime series.

        Returns one of: 'daily', 'weekly', 'monthly', 'yearly',
        'hourly', 'minutely', 'irregular'.
        """
        if len(dt_series) < 3:
            return "irregular"

        try:
            sorted_s = dt_series.sort_values()
            diffs = sorted_s.diff().dropna()
            median_diff = diffs.median().total_seconds()

            if median_diff < 120:
                return "minutely"
            if median_diff < 7200:
                return "hourly"
            if 72000 < median_diff < 100800:
                return "daily"
            if 561600 < median_diff < 691200:
                return "weekly"
            if 2419200 < median_diff < 2764800:
                return "monthly"
            if 30240000 < median_diff < 33696000:
                return "yearly"
            return "irregular"
        except Exception:
            return "irregular"

    # ── sample values ─────────────────────────────────────────────

    @staticmethod
    def _sample_values(series: pd.Series, n: int = 5) -> list[Any]:
        """Return up to *n* representative non-null values, JSON-safe."""
        clean = series.dropna()
        if len(clean) == 0:
            return []
        sampled = clean.sample(min(n, len(clean)), random_state=42)
        result: list[Any] = []
        for v in sampled:
            if isinstance(v, (np.integer,)):
                result.append(int(v))
            elif isinstance(v, (np.floating,)):
                result.append(round(float(v), 6))
            elif isinstance(v, (np.bool_,)):
                result.append(bool(v))
            elif isinstance(v, pd.Timestamp):
                result.append(v.isoformat())
            else:
                result.append(str(v))
        return result

    # ── dataset-level flags ───────────────────────────────────────

    def _build_dataset_flags(
        self,
        df: pd.DataFrame,
        col_profiles: list[ColumnProfile],
        n_rows: int,
        n_cols: int,
    ) -> DatasetFlags:
        """Aggregate column profiles and DataFrame into dataset-level flags."""
        id_cols = [
            cp.column_name for cp in col_profiles
            if cp.inferred_semantic_type == SemanticType.ID
        ]
        target = self._guess_target(df, col_profiles)

        dt_profiles = [
            cp for cp in col_profiles
            if cp.inferred_semantic_type == SemanticType.DATETIME
        ]
        has_dt_index = self._check_datetime_index(df, dt_profiles)
        is_ts = bool(has_dt_index or len(dt_profiles) > 0)

        task, imbalance = self._infer_task_type(df, target, col_profiles, is_ts)

        corr_pairs = self._compute_correlations(df)
        leakage = self._detect_leakage(df, col_profiles, target, corr_pairs)

        dup_count = int(df.duplicated().sum())
        mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 4)

        return DatasetFlags(
            num_rows=n_rows,
            num_columns=n_cols,
            is_time_series=is_ts,
            has_datetime_index=has_dt_index,
            suspected_id_columns=id_cols,
            suspected_target_column=target,
            likely_task_type=task,
            class_imbalance_ratio=imbalance,
            high_correlation_pairs=corr_pairs,
            data_leakage_risks=leakage,
            duplicate_row_count=dup_count,
            duplicate_row_percentage=round(dup_count / max(n_rows, 1) * 100, 4),
            memory_usage_mb=mem_mb,
        )

    # ── target guessing ───────────────────────────────────────────

    @staticmethod
    def _guess_target(
        df: pd.DataFrame, col_profiles: list[ColumnProfile]
    ) -> str | None:
        """Guess which column is the ML target.

        Priority:
        1. Columns whose semantic type is TARGET_CANDIDATE.
           If multiple match, pick the one with the lowest cardinality
           (more likely to be a classification label than a continuous
           feature that happened to match a name pattern like 'price').
        2. Last non-ID column as fallback.
        3. None if the DataFrame is empty or all-ID.
        """
        candidates = [
            cp for cp in col_profiles
            if cp.inferred_semantic_type == SemanticType.TARGET_CANDIDATE
        ]
        if candidates:
            # Tiebreaker: lowest cardinality → most likely a label column
            best = min(candidates, key=lambda cp: cp.nunique)
            return best.column_name

        # Fallback: last non-ID column
        for cp in reversed(col_profiles):
            if cp.inferred_semantic_type != SemanticType.ID:
                return cp.column_name
        return None

    # ── datetime index check ──────────────────────────────────────

    @staticmethod
    def _check_datetime_index(
        df: pd.DataFrame, dt_profiles: list[ColumnProfile]
    ) -> bool:
        """Check if any datetime column could serve as a monotonic index."""
        for cp in dt_profiles:
            try:
                s = pd.to_datetime(df[cp.column_name], errors="coerce").dropna()
                if len(s) < 2:
                    continue
                if s.is_monotonic_increasing and s.is_unique:
                    return True
            except Exception:
                continue
        return False

    # ── task type inference ───────────────────────────────────────

    @staticmethod
    def _infer_task_type(
        df: pd.DataFrame,
        target: str | None,
        col_profiles: list[ColumnProfile],
        is_ts: bool,
    ) -> tuple[TaskType, float | None]:
        """Infer the likely ML task type from the target column.

        Returns:
            Tuple of (task_type, class_imbalance_ratio_or_None).
        """
        if is_ts:
            return TaskType.TIME_SERIES, None

        if target is None:
            return TaskType.CLUSTERING, None

        if target not in df.columns:
            return TaskType.UNKNOWN, None

        series = df[target]
        nunique = series.nunique()

        # Classification check
        if nunique == 2:
            vc = series.value_counts()
            ratio = round(float(vc.iloc[0] / vc.iloc[-1]), 4) if len(vc) == 2 else None
            return TaskType.BINARY_CLASSIFICATION, ratio
        if 3 <= nunique <= 30:
            if series.dtype == object or pd.api.types.is_integer_dtype(series):
                vc = series.value_counts()
                ratio = round(float(vc.iloc[0] / vc.iloc[-1]), 4) if len(vc) >= 2 else None
                return TaskType.MULTICLASS_CLASSIFICATION, ratio

        # Regression: numeric with many unique values
        if pd.api.types.is_numeric_dtype(series) and nunique > 15:
            return TaskType.REGRESSION, None

        return TaskType.UNKNOWN, None

    # ── correlations ──────────────────────────────────────────────

    @staticmethod
    def _compute_correlations(
        df: pd.DataFrame, threshold: float = 0.7
    ) -> list[CorrelationPair]:
        """Find all pairs of numeric columns with |correlation| > threshold.

        Computes both Pearson and Spearman. A pair is included if either
        method exceeds the threshold.
        """
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return []

        pairs: list[CorrelationPair] = []
        seen: set[tuple[str, str]] = set()

        try:
            pearson_corr = num_df.corr(method="pearson")
            spearman_corr = num_df.corr(method="spearman")
        except Exception:
            return []

        cols = num_df.columns.tolist()
        for i, ca in enumerate(cols):
            for cb in cols[i + 1:]:
                key = (ca, cb)
                if key in seen:
                    continue
                seen.add(key)

                p = pearson_corr.loc[ca, cb]
                s = spearman_corr.loc[ca, cb]

                if pd.isna(p):
                    p = None
                if pd.isna(s):
                    s = None

                flagged = None
                if p is not None and abs(p) > threshold:
                    flagged = CorrelationMethod.PEARSON
                if s is not None and abs(s) > threshold:
                    flagged = flagged or CorrelationMethod.SPEARMAN

                if flagged:
                    pairs.append(CorrelationPair(
                        column_a=ca,
                        column_b=cb,
                        pearson=round(p, 6) if p is not None else None,
                        spearman=round(s, 6) if s is not None else None,
                        method_flagged=flagged,
                    ))

        return pairs

    # ── leakage detection ─────────────────────────────────────────

    @staticmethod
    def _detect_leakage(
        df: pd.DataFrame,
        col_profiles: list[ColumnProfile],
        target: str | None,
        corr_pairs: list[CorrelationPair],
    ) -> list[DataLeakageRisk]:
        """Flag columns that look like post-hoc data leakage risks."""
        risks: list[DataLeakageRisk] = []

        # Name-based heuristic
        for cp in col_profiles:
            if _POSTHOC_PAT.search(cp.column_name):
                risks.append(DataLeakageRisk(
                    column_name=cp.column_name,
                    reason=LeakageReason.NAME_SUGGESTS_POST_HOC,
                    detail=f"Column name '{cp.column_name}' suggests post-hoc data.",
                ))

        # Correlation-based: near-perfect correlation with target
        if target:
            for pair in corr_pairs:
                other = None
                if pair.column_a == target:
                    other = pair.column_b
                elif pair.column_b == target:
                    other = pair.column_a

                if other is None:
                    continue

                p = abs(pair.pearson) if pair.pearson is not None else 0
                if p > 0.95:
                    reason = (LeakageReason.PERFECT_CORRELATION_WITH_TARGET
                              if p > 0.99
                              else LeakageReason.NEAR_PERFECT_CORRELATION_WITH_TARGET)
                    risks.append(DataLeakageRisk(
                        column_name=other,
                        reason=reason,
                        detail=f"|Pearson| = {p:.4f} with target '{target}'.",
                    ))

        return risks
