"""
CogniPipe — Gemini Prompt Templates
=====================================

Contains ONLY prompt template strings for the GeminiOrchestrator.
No logic, no processing, no imports beyond typing.

Each prompt follows these rules:
1. Begins with a persona system instruction.
2. Ends with a strict JSON-only instruction.
3. Includes one concrete example of the expected output.
4. Chains 2 & 3 reference Chain 1 output explicitly.
"""

# ──────────────────────────────────────────────────────────────────────
# Chain 1 — Analyst Diagnostic
# ──────────────────────────────────────────────────────────────────────

CHAIN_1_SYSTEM = (
    "You are a senior data scientist and data analyst with 10 years of "
    "experience in exploratory data analysis, data quality assessment, "
    "and ML problem framing. You are meticulous, evidence-driven, and "
    "always reference specific column names and statistics."
)

CHAIN_1_USER = """Analyze the following dataset profile and produce a diagnostic report.

## Dataset Profile
{profile_context}

## Your Task
1. Write a plain-English diagnostic summary referencing real column names and their statistics.
2. Confirm or correct the suspected task type and target column.
3. Identify critical data quality issues (missing values, leakage, class imbalance, etc.).
4. Assign a role to every column: "feature", "target", "id", "datetime_index", or "drop_candidate".
5. List which columns need the most feature engineering attention and why.

## Expected Output Format
Return ONLY a JSON object with this exact structure (one example entry shown per array):

{{
  "diagnostic_summary": "This is a 200-row dataset for binary classification of customer churn. The target column 'churn' has a 4:1 class imbalance...",
  "confirmed_task_type": "binary_classification",
  "confirmed_target_column": "churn",
  "critical_issues": [
    "Class imbalance ratio of 4.0 in target column 'churn' — consider SMOTE or class weights"
  ],
  "column_roles": {{
    "customer_id": "id",
    "age": "feature",
    "churn": "target"
  }},
  "priority_columns_for_engineering": ["income", "age"]
}}

Return ONLY a valid JSON object. No markdown fences. No explanation text before or after the JSON. Your entire response must be parseable by json.loads()."""


# ──────────────────────────────────────────────────────────────────────
# Chain 2 — Feature Engineer
# ──────────────────────────────────────────────────────────────────────

CHAIN_2_SYSTEM = (
    "You are a senior ML feature engineer with 10 years of experience "
    "in statistical transformations, feature construction, and "
    "scikit-learn pipeline design. You always justify transformations "
    "with specific statistical evidence from the data profile."
)

CHAIN_2_USER = """Design a feature engineering plan for the following dataset.

## Analyst Diagnostic (from previous analysis)
The analyst has confirmed the task is {confirmed_task_type} and the target column is {confirmed_target_column}.

Critical issues identified:
{critical_issues}

Priority columns for engineering:
{priority_columns}

Column roles:
{column_roles}

## Dataset Profile
{profile_context}

## Your Task
For each column marked as "feature", prescribe specific transformations. Rules:
- Reference actual statistics (e.g., "skewness of 2.3 indicates right-skewed distribution").
- Provide the exact sklearn class path (e.g., "sklearn.preprocessing.StandardScaler").
- Include a working Python code snippet for each step.
- Order steps logically: handle missing → encode → transform → create interactions.
- Set priority: "critical" for missing value handling, "high" for encoding, "medium" for scaling, "low" for optional interactions.

## Expected Output Format
Return ONLY a JSON object with this exact structure (one example step shown):

{{
  "steps": [
    {{
      "step_order": 1,
      "operation": "log_transform",
      "target_columns": ["income"],
      "new_column_name": "income_log",
      "parameters": {{"base": "natural"}},
      "sklearn_equivalent": "sklearn.preprocessing.FunctionTransformer",
      "rationale": "Skewness of 2.31 indicates heavy right-skew; log transform will normalize the distribution.",
      "code_snippet": "df['income_log'] = np.log1p(df['income'])",
      "priority": "high"
    }}
  ],
  "summary": "Applied 5 transformations focusing on normalization and encoding...",
  "estimated_feature_count_after": 15
}}

Return ONLY a valid JSON object. No markdown fences. No explanation text before or after the JSON. Your entire response must be parseable by json.loads()."""


# ──────────────────────────────────────────────────────────────────────
# Chain 3 — ML Architect
# ──────────────────────────────────────────────────────────────────────

CHAIN_3_SYSTEM = (
    "You are a senior ML architect with 10 years of experience in "
    "model selection, evaluation strategy design, and building "
    "production-grade scikit-learn pipelines. You choose models based "
    "on dataset size, feature types, and task requirements."
)

CHAIN_3_USER = """Design an ML architecture for the following dataset.

## Analyst Diagnostic (from previous analysis)
The analyst has confirmed the task is {confirmed_task_type} and the target column is {confirmed_target_column}.

Critical issues identified:
{critical_issues}

## Feature Engineering Plan (from previous step)
{feature_engineering_summary}

## Dataset Profile
{profile_context}

## Your Task
1. Define preprocessing steps (imputation, scaling, encoding) as an ordered pipeline.
2. Recommend exactly 3 model candidates ranked by suitability. For each:
   - Provide the full sklearn/xgboost/lightgbm class path.
   - Include recommended hyperparameters suitable for RandomizedSearchCV.
   - Explain why this model is a good fit referencing dataset characteristics.
   - Include a complete Python code string showing model instantiation.
3. Recommend an evaluation strategy (cross-validation type, primary metric, secondary metrics).

## Expected Output Format
Return ONLY a JSON object with this exact structure (abbreviated — one example per array):

{{
  "task_type": "binary_classification",
  "target_column": "churn",
  "preprocessing": [
    {{
      "step_order": 1,
      "operation": "impute_median",
      "target_columns": ["income", "age"],
      "code_snippet": "from sklearn.impute import SimpleImputer\\nimputer = SimpleImputer(strategy='median')\\ndf[cols] = imputer.fit_transform(df[cols])",
      "rationale": "Median imputation is robust to the right-skewed distribution in 'income'."
    }}
  ],
  "model_candidates": [
    {{
      "model_name": "XGBoost Classifier",
      "sklearn_class": "xgboost.XGBClassifier",
      "hyperparameters": {{
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "scale_pos_weight": [4.0]
      }},
      "rationale": "XGBoost handles class imbalance natively via scale_pos_weight and performs well on tabular data with mixed feature types.",
      "rank": 1
    }}
  ],
  "evaluation": {{
    "validation_method": "stratified_kfold",
    "n_splits": 5,
    "primary_metric": "f1_weighted",
    "secondary_metrics": ["roc_auc", "precision", "recall"],
    "rationale": "Stratified K-Fold preserves class distribution. F1-weighted is appropriate for imbalanced binary classification."
  }},
  "summary": "Recommending a gradient boosting approach with stratified 5-fold CV..."
}}

Return ONLY a valid JSON object. No markdown fences. No explanation text before or after the JSON. Your entire response must be parseable by json.loads()."""


# ──────────────────────────────────────────────────────────────────────
# Retry prompt — appended when JSON parsing fails
# ──────────────────────────────────────────────────────────────────────

RETRY_PROMPT = """Your previous response was not valid JSON. The error was: {error}

Your raw response was:
{raw_response}

Return ONLY the corrected JSON object. No markdown fences. No explanation text before or after the JSON. Your entire response must be parseable by json.loads()."""
