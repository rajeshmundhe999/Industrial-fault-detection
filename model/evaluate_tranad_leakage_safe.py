"""
TranAD Leakage-Safe Evaluation
================================
Evaluates the existing TranAD checkpoint under the same strict protocol
as GRU-AE v2 (val_max threshold, batch-aware, no test labels used).

TranAD forward: returns (x1, x2) — two reconstruction stages.
Score per window = mean of (x1-src)^2 and (x2-src)^2, then robust-max
over sensors at the end timestep (same scoring as GRU-AE v2).

Outputs → model/results/tranad_leakage_safe/
  metrics.csv
"""
import csv, os, sys
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_fscore_support)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
CKPT  = os.path.join(BASE, "model", "saved", "tranad_original_best.pt")
OUT   = os.path.join(BASE, "model", "results", "tranad_leakage_safe")
sys.path.insert(0, BASE)
from model.tranad_original import TranADOriginal

os.makedirs(OUT, exist_ok=True)

WINDOW = 60; STRIDE = 5
SENSORS = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
           "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]


def scale(x, mn, sc): return x * sc + mn

def make_split(batch_data, faults, batches, mn, sc):
    xs, y_end, batch_ids, times = [], [], [], []
    for b in batches:
        x = scale(batch_data[str(b)], mn, sc)
        f = faults[str(b)]["fault"]
        t = faults[str(b)]["time"]
        for start in range(0, len(x) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end-1] > 0))
            batch_ids.append(b); times.append(float(t[end-1]))
    return (np.asarray(xs), np.asarray(y_end),
            np.asarray(batch_ids), np.asarray(times))

def get_errors(model, x_np, device, bs=256):
    """Returns (N, T, S) averaged reconstruction errors from both TranAD heads."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x_np), bs):
            xb  = torch.tensor(x_np[i:i+bs], dtype=torch.float32).to(device)
            # TranAD expects (T, B, S) — permute
            src = xb.permute(1, 0, 2)   # (T, B, S)
            tgt = src[-1:].expand_as(src)  # use last timestep as tgt
            x1, x2 = model(src, tgt)
            # Average both heads, permute back to (B, T, S)
            err = 0.5 * ((x1 - src)**2 + (x2 - src)**2)
            err = err.permute(1, 0, 2).cpu().numpy()  # (B, T, S)
            chunks.append(err)
    return np.concatenate(chunks)

def robust_max_score(errors, median, mad):
    end_err = errors[:, -1, :]
    z       = (end_err - median) / (mad + 1e-12)
    return z.max(axis=1)

def event_recall(pred, batch_ids, faulty):
    return sum(1 for b in faulty if pred[batch_ids == b].any()) / len(faulty)

def metrics_dict(y, pred, score, batch_ids, faulty):
    roc = roc_auc_score(y, score)
    ap  = average_precision_score(y, score)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    er  = event_recall(pred, batch_ids, faulty)
    return dict(roc_auc=round(roc,4), avg_prec=round(ap,4),
                f1=round(f,4), recall=round(r,4), precision=round(p,4),
                event_recall=round(er,4))


def main():
    print("=" * 60)
    print("TranAD LEAKAGE-SAFE EVALUATION")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TranADOriginal(feats=9).to(device)
    # handle possible state_dict key
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TranAD params: {n_params:,}")

    # ── data ──────────────────────────────────────────────────────────────────
    batch_data = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
    sc_mn      = np.load(os.path.join(DATA, "scaler_min.npy")).astype(np.float32)
    sc_sc      = np.load(os.path.join(DATA, "scaler_scale.npy")).astype(np.float32)

    faulty_batches = list(range(91, 101))
    xtr, _,   _,   _   = make_split(batch_data, faults, list(range(1, 71)),  sc_mn, sc_sc)
    xva, _,   _,   _   = make_split(batch_data, faults, list(range(71, 81)), sc_mn, sc_sc)
    xte, yte, bte, tte = make_split(batch_data, faults, list(range(81, 101)),sc_mn, sc_sc)
    print(f"Train={xtr.shape}  Val={xva.shape}  Test={xte.shape}")

    print("Computing errors (train)...")
    etr = get_errors(model, xtr, device)
    print("Computing errors (val)...")
    eva = get_errors(model, xva, device)
    print("Computing errors (test)...")
    ete = get_errors(model, xte, device)

    # ── scoring ───────────────────────────────────────────────────────────────
    train_end  = etr[:, -1, :]
    median     = np.median(train_end, axis=0)
    mad        = np.median(np.abs(train_end - median), axis=0) * 1.4826 + 1e-12

    val_scores  = robust_max_score(eva, median, mad)
    test_scores = robust_max_score(ete, median, mad)
    threshold   = float(val_scores.max())
    pred        = (test_scores > threshold).astype(int)

    m = metrics_dict(yte, pred, test_scores, bte, faulty_batches)
    print(f"\nTranAD (leakage-safe val_max):")
    print(f"  AUC={m['roc_auc']}  F1={m['f1']}  Recall={m['recall']}  "
          f"Precision={m['precision']}  EvRec={m['event_recall']}")
    print(f"  Threshold={threshold:.4f}")

    # ── per-batch breakdown ───────────────────────────────────────────────────
    print("\nPer-batch:")
    for b in faulty_batches:
        ix  = bte == b
        det = int(pred[ix & (yte == 1)].any())
        print(f"  B{b}: detected={'YES' if det else 'NO'}  "
              f"alarms={int(pred[ix].sum())}  fault_wins={int(yte[ix].sum())}")

    # ── save ──────────────────────────────────────────────────────────────────
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in m.items():
            w.writerow([k, v])
        w.writerow(["threshold", round(threshold, 4)])
        w.writerow(["n_params", n_params])

    np.save(os.path.join(OUT, "test_scores.npy"),  test_scores)
    np.save(os.path.join(OUT, "val_scores.npy"),   val_scores)
    np.save(os.path.join(OUT, "test_labels.npy"),  yte)
    np.save(os.path.join(OUT, "test_batch_ids.npy"), bte)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
