"""One-command entry point for the reproducible PFDI pipeline."""
import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def run(script):
    subprocess.run([sys.executable, os.path.join(BASE, script)], check=True)


def main():
    p = argparse.ArgumentParser(description="PFDI project runner")
    p.add_argument("command", choices=["prepare", "train", "evaluate", "baselines"],
                   help="pipeline stage to run")
    args = p.parse_args()
    stages = {
        "prepare": ["data/loader.py", "data/preprocess.py", "model/build_aligned_dataset.py"],
        "train": ["model/train_gru.py"],
        "evaluate": ["model/leakage_safe_event_evaluation.py"],
        "baselines": ["model/leakage_safe_pca_baselines.py"],
    }
    for script in stages[args.command]: run(script)


if __name__ == "__main__":
    main()
