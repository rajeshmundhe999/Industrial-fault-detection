"""Causal batch monitor for the validated GRU-AE operating point.

Example:
    python model/monitor.py --batch-id 91
    python model/monitor.py --input path/to/raw_9_sensor_array.npy
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from model.gru_ae import GRUAutoencoder

ARTIFACT = os.path.join(BASE, "model", "results", "leakage_safe", "operating_point.npz")
MODEL = os.path.join(BASE, "model", "saved", "gru_ae_best.pt")
DATA = os.path.join(BASE, "data", "processed")


def load_monitor():
    """Load the frozen model and calibration artifacts."""
    artifact = np.load(ARTIFACT)
    model = GRUAutoencoder(input_size=9)
    model.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=False)["model_state_dict"])
    model.eval()
    return model, artifact


def score_batch(raw_data, times=None):
    """Return causal alarms for a raw (timesteps, 9) sensor array.

    No fault labels are expected or accepted.  The first score is emitted only
    once 60 samples are available; windows advance every five samples.
    """
    model, a = load_monitor()
    raw_data = np.asarray(raw_data, dtype=np.float32)
    if raw_data.ndim != 2 or raw_data.shape[1] != 9:
        raise ValueError("raw_data must have shape (timesteps, 9)")
    if len(raw_data) < int(a["window_size"]):
        return []
    scaled = raw_data * a["scaler_scale"] + a["scaler_min"]
    window, stride = int(a["window_size"]), int(a["stride"])
    records = []
    with torch.no_grad():
        for start in range(0, len(scaled) - window + 1, stride):
            x = torch.tensor(scaled[start:start + window][None], dtype=torch.float32)
            error = ((model(x) - x) ** 2).numpy()[0, -1]
            standardized = (error - a["sensor_median"]) / a["sensor_robust_scale"]
            score = float(np.max(standardized))
            sensor_index = int(np.argmax(standardized))
            records.append({
                "end_index": start + window - 1,
                "time": None if times is None else float(times[start + window - 1]),
                "score": score,
                "threshold": float(a["threshold"]),
                "alarm": bool(score > a["threshold"]),
                "top_sensor_index": sensor_index,
                "per_sensor_standardized_error": standardized.tolist(),
            })
    return records


def main():
    p = argparse.ArgumentParser(description="Run the leakage-safe GRU batch monitor")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch-id", type=int, help="processed IndPenSim batch to score")
    group.add_argument("--input", help="raw .npy array with shape (timesteps, 9)")
    p.add_argument("--output", help="optional JSON output path")
    args = p.parse_args()
    if args.batch_id is not None:
        d = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
        f = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
        raw, times = d[str(args.batch_id)], f[str(args.batch_id)]["time"]
    else:
        raw, times = np.load(args.input), None
    records = score_batch(raw, times)
    payload = {"n_windows": len(records), "n_alarms": sum(r["alarm"] for r in records), "records": records}
    text = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w") as f: f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
