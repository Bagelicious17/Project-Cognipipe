# smoke_test.py — run this from your project root
import pandas as pd
import os
from dotenv import load_dotenv
from services.data_profiler import DataProfiler

load_dotenv()  # Load .env file so os.getenv() can find GEMINI_API_KEY
from services.gemini_orchestrator import GeminiOrchestrator
from services.code_assembler import CodeAssembler

# Download Titanic if you don't have it
# https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

df = pd.read_csv("titanic.csv")

print("=" * 60)
print("LAYER 1 — Profiling")
print("=" * 60)
profiler = DataProfiler()
profile = profiler.profile(df)
print(f"Columns profiled: {len(profile.columns)}")
print(f"Task type detected: {profile.dataset.likely_task_type}")
print(f"Suspected target: {profile.dataset.suspected_target_column}")
print(f"Suspected IDs: {profile.dataset.suspected_id_columns}")
print(f"Leakage risks: {[r.column_name for r in profile.dataset.data_leakage_risks]}")
print()

print("=" * 60)
print("LAYER 2 — Gemini Analysis")
print("=" * 60)
orchestrator = GeminiOrchestrator(
    api_key=os.getenv("GEMINI_API_KEY"),
    model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
)
gemini_result = orchestrator.run(profile)
print(f"Confirmed task: {gemini_result.analyst_diagnostic.confirmed_task_type}")
print(f"Confirmed target: {gemini_result.analyst_diagnostic.confirmed_target_column}")
print(f"Critical issues: {gemini_result.analyst_diagnostic.critical_issues}")
print(f"Feature steps: {len(gemini_result.feature_engineering.steps)}")
print(f"Model candidates: {len(gemini_result.ml_architecture.model_candidates)}")
print()
print("Token usage:")
for usage in gemini_result.token_usage:
    print(f"  {usage.chain_name}: {usage.total_tokens} tokens")
print()

print("=" * 60)
print("LAYER 3 — Code Assembly")
print("=" * 60)
assembler = CodeAssembler()
pipeline = assembler.build(profile, gemini_result)

with open("smoke_output/pipeline.py", "w") as f:
    f.write(pipeline.python_script)
print("pipeline.py written")

if pipeline.notebook_json:
    with open("smoke_output/pipeline.ipynb", "w") as f:
        f.write(pipeline.notebook_json)
    print("pipeline.ipynb written")

with open("smoke_output/requirements.txt", "w") as f:
    f.write(pipeline.requirements_txt)
print("requirements.txt written")

print()
print("requirements.txt contents:")
print(pipeline.requirements_txt)