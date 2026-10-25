"""
Full pipeline runner: evaluate v2 model then run SHAP XAI.
Run this after training completes.
"""
import os, sys, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

def run(script, desc):
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print('='*60)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, os.path.join(MODEL_DIR, script)],
        cwd=BASE, env=env
    )
    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}")
        return False
    return True

if __name__ == "__main__":
    ok = run("evaluate_gru_v2.py", "GRU-AE v2 Leakage-Safe Evaluation")
    if ok:
        run("xai_analysis.py", "SHAP Fault Isolation Analysis")
    print("\nPipeline complete.")
