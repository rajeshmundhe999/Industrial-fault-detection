"""Paper-aligned GRU-AE benchmark reproduction.

This is intentionally a separate protocol from leakage_safe_event_evaluation:
86 normal batches train, batch 87 calibrates the threshold, batches 91--100
test. It reports per-fault-batch detection rate (FDR), the metric reported by
Petrovski et al. (2023), and normal false-alarm rates alongside it.

The original paper does not provide every preprocessing/architecture detail in
the article. This is therefore a transparent *paper-like reproduction*, not an
assertion of exact replication. It uses 25 dense process variables (all process
columns excluding label/control metadata and columns with >50% missingness),
normal-only min-max scaling, shuffled normal training sequences, and its stated
median + 1.5 IQR threshold rule.
"""
import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw", "100_Batches_IndPenSim_V3.csv")
OUT = os.path.join(BASE, "model", "results", "paper_protocol")
SAVED = os.path.join(BASE, "model", "saved", "paper_protocol_gru.pt")
SEED, WINDOW, EPOCHS, BATCH_SIZE = 42, 60, 30, 256


class PaperGRUAE(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.e1 = nn.GRU(n_features, 32, batch_first=True)
        self.e2 = nn.GRU(32, 12, batch_first=True)
        self.e3 = nn.GRU(12, 4, batch_first=True)
        self.d1 = nn.GRU(4, 12, batch_first=True)
        self.d2 = nn.GRU(12, 32, batch_first=True)
        self.out = nn.Linear(32, n_features)
    def forward(self, x):
        x, _ = self.e1(x); x, _ = self.e2(x); x, _ = self.e3(x)
        x, _ = self.d1(x); x, _ = self.d2(x)
        return self.out(x)


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def load_data():
    # The source CSV has trailing empty columns; reading the first 39 fields is
    # deliberate.  The first 31 are time + 30 process variables + fault label.
    df = pd.read_csv(RAW, usecols=list(range(39)), low_memory=False)
    batch_col, fault_col = df.columns[35], df.columns[31]
    process_cols = list(df.columns[1:31])
    normal = df[df[batch_col].between(1, 90)]
    dense = [c for c in process_cols if normal[c].isna().mean() <= 0.50]
    # Median imputation is learned exclusively from normal training batches.
    train_normal = df[df[batch_col].between(1, 86)]
    medians = train_normal[dense].median()
    x = df[dense].fillna(medians).to_numpy(np.float32)
    return df, x, dense, batch_col, fault_col


def windows(x, window=WINDOW, stride=1):
    return np.asarray([x[i:i + window] for i in range(0, len(x) - window + 1, stride)], dtype=np.float32)


def scores(model, x):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), 256):
            z = torch.tensor(x[i:i + 256], dtype=torch.float32)
            # Window-end MSE gives a causal point-level fault decision.
            out.append(((model(z)[:, -1] - z[:, -1]) ** 2).mean(1).numpy())
    return np.concatenate(out)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    seed_everything(); os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.dirname(SAVED), exist_ok=True)
    df, raw_x, features, batch_col, fault_col = load_data()
    train_rows = df[batch_col].between(1, 86).to_numpy()
    scaler = MinMaxScaler().fit(raw_x[train_rows])
    x = scaler.transform(raw_x).astype(np.float32)
    # Shuffle normal *time points* before making training sequences, matching
    # the paper's stated distribution-discrepancy strategy.
    shuffled = x[train_rows].copy(); np.random.default_rng(SEED).shuffle(shuffled)
    train_x = windows(shuffled, stride=WINDOW)
    val_x = windows(x[df[batch_col].eq(87).to_numpy()])
    model = PaperGRUAE(len(features))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(); loader = DataLoader(TensorDataset(torch.tensor(train_x)), batch_size=BATCH_SIZE, shuffle=True)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for (z,) in loader:
            recon = model(z); loss = loss_fn(recon, z)
            optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
            total += loss.item() * len(z)
        with torch.no_grad():
            val_loss = loss_fn(model(torch.tensor(val_x)), torch.tensor(val_x)).item()
        train_loss = total / len(train_x); history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d}: train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best - 1e-7:
            best, stale = val_loss, 0; torch.save({"state": model.state_dict(), "features": features}, SAVED)
        else: stale += 1
        if stale >= 12: break
    model.load_state_dict(torch.load(SAVED, map_location="cpu", weights_only=False)["state"])
    val_scores = scores(model, val_x)
    q1, q3 = np.percentile(val_scores, [25, 75]); threshold = float(np.median(val_scores) + 1.5 * (q3 - q1))
    batch_rows = []
    for batch in range(88, 101):
        mask = df[batch_col].eq(batch).to_numpy(); xb, faults = x[mask], df.loc[mask, fault_col].to_numpy()
        sb = scores(model, windows(xb)); endpoint_fault = faults[WINDOW - 1:]
        alarm = sb > threshold
        # Paper-comparable FDR: fraction of fault-active samples detected.
        if batch >= 91:
            active = endpoint_fault > 0
            fdr = float(alarm[active].mean()) if active.any() else np.nan
            batch_rows.append({"batch": batch, "kind": "fault", "active_points": int(active.sum()), "fdr": fdr,
                               "all_window_alarm_rate": float(alarm.mean()), "first_alarm_offset": int(np.flatnonzero(alarm & active)[0]) if np.any(alarm & active) else None})
        else:
            batch_rows.append({"batch": batch, "kind": "normal", "active_points": 0, "fdr": None,
                               "all_window_alarm_rate": float(alarm.mean()), "first_alarm_offset": None})
    fdrs = [r["fdr"] for r in batch_rows if r["kind"] == "fault"]
    normal_fa = [r["all_window_alarm_rate"] for r in batch_rows if r["kind"] == "normal"]
    write_csv(os.path.join(OUT, "per_batch_fdr.csv"), batch_rows); write_csv(os.path.join(OUT, "training_history.csv"), history)
    summary = {"protocol": "paper_like_gru_shuffle", "n_features": len(features), "features": features, "threshold": threshold,
               "threshold_rule": "validation batch 87 median + 1.5 IQR", "mean_fault_detection_rate": float(np.mean(fdrs)),
               "per_fault_fdr": fdrs, "normal_batches_88_90_mean_alarm_rate": float(np.mean(normal_fa)), "best_val_loss": best}
    with open(os.path.join(OUT, "summary.json"), "w") as f: json.dump(summary, f, indent=2)
    print("FEATURES", len(features)); print("THRESHOLD", threshold); print("MEAN FDR", summary["mean_fault_detection_rate"]); print("NORMAL FPR", summary["normal_batches_88_90_mean_alarm_rate"])


if __name__ == "__main__": main()
