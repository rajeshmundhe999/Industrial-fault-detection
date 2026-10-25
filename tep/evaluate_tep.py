"""
Evaluate GRU-AE v2 on the TEP benchmark.

Scoring follows the leakage-safe evaluate_gru_v2.py protocol:
  - Reconstruction error per window per sensor
  - Standardize with training-set median/MAD (robust_max scoring)
  - Threshold = max validation score (val_max policy, zero test labels used)
  - Reports: ROC-AUC, F1, Recall, Precision per overall + per fault type
  - Saves: tep/results/metrics.csv
"""

import csv, os, sys
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, average_precision_score
)

TEP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE    = os.path.dirname(TEP_DIR)
DATA    = os.path.join(TEP_DIR, "processed")
SAVED   = os.path.join(TEP_DIR, "saved", "gru_ae_v2_tep_best.pt")
OUT     = os.path.join(TEP_DIR, "results")
sys.path.insert(0, BASE)
os.makedirs(OUT, exist_ok=True)

from model.gru_ae_v2 import GRUAutoencoderV2

WINDOW      = 60
STRIDE      = 5
N_SENSORS   = 22
FAULT_STEP  = 160
SKIP_FAULTS = {3, 9}

# Fault-id to batch-id mapping (same as preprocess_tep.py)
# batch_id 0 = normal (d00_te), batch_ids 1.. = fault IDs in order
FAULT_IDS = [0] + [k for k in range(1, 22) if k not in SKIP_FAULTS]


def reconstruction_errors(model, x, device):
    """Returns (N, T, S) squared reconstruction errors."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            z = torch.tensor(x[start:start + 512], dtype=torch.float32).to(device)
            r = model(z)
            chunks.append(((r - z) ** 2).cpu().numpy())
    return np.concatenate(chunks)


def robust_max_score(errors, median, mad):
    """
    Window-end robust-max score:
      1. Standardise each sensor error at window-end with median/MAD
      2. Take the maximum across sensors → scalar score per window
    """
    end_err = errors[:, -1, :]                    # (N, S)
    z       = (end_err - median) / (mad + 1e-12)  # (N, S)
    return z.max(axis=1)                           # (N,)


def main():
    print("=" * 60)
    print("TEP EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ── load data ─────────────────────────────────────────────────────────────
    X_train      = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    X_val        = np.load(os.path.join(DATA, "X_val.npy")).astype(np.float32)
    X_test       = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)
    y_test       = np.load(os.path.join(DATA, "y_test.npy"))
    batch_ids    = np.load(os.path.join(DATA, "batch_ids_test.npy"))

    print(f"X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")
    print(f"Fault windows: {y_test.sum()}/{len(y_test)}")

    # ── load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(SAVED, map_location="cpu", weights_only=False)
    model = GRUAutoencoderV2(input_size=N_SENSORS, dropout=0.15)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', 0):.6f}")

    # ── compute reconstruction errors ─────────────────────────────────────────
    print("Computing errors (train)...")
    etr = reconstruction_errors(model, X_train, device)   # (Ntr, T, S)
    print("Computing errors (val)...")
    eva = reconstruction_errors(model, X_val,   device)   # (Nva, T, S)
    print("Computing errors (test)...")
    ete = reconstruction_errors(model, X_test,  device)   # (Nte, T, S)

    # ── training-set statistics for normalisation ─────────────────────────────
    train_end = etr[:, -1, :]                               # (Ntr, S)
    median    = np.median(train_end, axis=0)                # (S,)
    mad       = np.median(np.abs(train_end - median), axis=0) * 1.4826 + 1e-12

    # ── scores ────────────────────────────────────────────────────────────────
    val_scores  = robust_max_score(eva, median, mad)        # (Nva,)
    test_scores = robust_max_score(ete, median, mad)        # (Nte,)

    # Threshold = max normal validation score (val_max policy)
    threshold = float(val_scores.max())
    print(f"\nThreshold (val_max): {threshold:.4f}")

    pred = (test_scores > threshold).astype(int)

    # ── overall metrics ───────────────────────────────────────────────────────
    roc_auc = roc_auc_score(y_test, test_scores)
    ap      = average_precision_score(y_test, test_scores)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0)

    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Avg Prec  : {ap:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Threshold : {threshold:.4f}")

    # ── per-fault-type metrics ────────────────────────────────────────────────
    print("\nPER-FAULT DETECTION RATE  (batch_id → fault_id → DR)")
    print("-" * 55)
    fault_rows = []
    for bid in np.unique(batch_ids):
        fid  = FAULT_IDS[bid]
        ix   = np.flatnonzero(batch_ids == bid)
        y_b  = y_test[ix]
        sc_b = test_scores[ix]
        pr_b = pred[ix]

        total_fault = int(y_b.sum())
        detected    = int(pr_b[y_b == 1].sum())
        dr          = detected / total_fault if total_fault > 0 else float("nan")

        auc_b = roc_auc_score(y_b, sc_b) if len(np.unique(y_b)) == 2 else float("nan")

        tag = f"F{fid:02d}" if fid != 0 else "Normal"
        print(f"  batch={bid:2d}  fault={tag}  "
              f"fault_wins={total_fault:4d}  detected={detected:4d}  "
              f"DR={dr:.3f}  AUC={auc_b:.4f}" if not np.isnan(auc_b)
              else f"  batch={bid:2d}  fault={tag}  "
                   f"fault_wins={total_fault:4d}  detected={detected:4d}  DR={dr}")

        fault_rows.append(dict(
            batch_id=int(bid), fault_id=fid,
            fault_windows=total_fault, detected=detected,
            detection_rate=round(dr, 4) if not np.isnan(dr) else "nan",
            roc_auc=round(auc_b, 4) if not np.isnan(auc_b) else "nan",
        ))

    # ── save results ──────────────────────────────────────────────────────────
    summary = dict(
        roc_auc=round(roc_auc, 4),
        average_precision=round(ap, 4),
        f1=round(f1, 4),
        recall=round(rec, 4),
        precision=round(prec, 4),
        threshold=round(threshold, 4),
        n_test_windows=int(len(y_test)),
        n_fault_windows=int(y_test.sum()),
    )

    # Write metrics.csv with one summary row + per-fault rows
    metrics_path = os.path.join(OUT, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        # Summary section
        w = csv.writer(f)
        w.writerow(["section", "metric", "value"])
        for k, v in summary.items():
            w.writerow(["overall", k, v])
        w.writerow([])
        # Per-fault section
        fault_fields = list(fault_rows[0].keys())
        w2 = csv.DictWriter(f, fieldnames=["section"] + fault_fields)
        w2.writeheader()
        for row in fault_rows:
            w2.writerow({"section": "per_fault", **row})

    # Also save scores
    np.save(os.path.join(OUT, "test_scores.npy"),  test_scores)
    np.save(os.path.join(OUT, "val_scores.npy"),   val_scores)
    np.save(os.path.join(OUT, "threshold.npy"),    np.array(threshold))
    np.save(os.path.join(OUT, "predictions.npy"),  pred)

    print(f"\nResults saved to: {OUT}")
    print("metrics.csv written.")


if __name__ == "__main__":
    main()
