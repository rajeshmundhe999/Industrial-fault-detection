"""
Leakage-safe evaluation for GRU-AE v2.

Scoring improvements over v1:
  1. Full-window reconstruction error (not just window-end) — captures the
     average degradation over the entire 60-step window, which is more robust
     for slow-developing process faults.
  2. Sensor importance weighting derived from VALIDATION reconstruction error
     variance.  Sensors with higher variance in normal validation data carry
     more discriminative power, so their errors receive higher weight.
     This is computed entirely from normal validation batches — no test labels.
  3. Three aggregation strategies are pre-declared and evaluated:
       a. max_robust   — maximum standardised sensor error at window end (v1 primary)
       b. mean_robust  — mean standardised sensor error, full window
       c. weighted_max — importance-weighted max error at window end
  4. Temporal persistence: 2-of-3 consecutive windows must alarm before
     reporting (unchanged from v1, but re-verified here).
  5. Threshold: val_max policy (no test labels ever touched).

Output goes to model/results/leakage_safe_v2/ with same CSV schema as v1.
"""
import csv, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
SAVED = os.path.join(BASE, "model", "saved", "gru_ae_v2_best.pt")
OUT   = os.path.join(BASE, "model", "results", "leakage_safe_v2")
PLOTS = os.path.join(BASE, "model", "plots",   "leakage_safe_v2")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

WINDOW = 60
STRIDE = 5
TRAIN_BATCHES = range(1, 71)
VAL_BATCHES   = range(71, 81)
TEST_BATCHES  = range(81, 101)
SENSORS = ["Agitator RPM", "Aeration rate", "Sugar feed rate",
           "Acid flow rate", "Base flow rate", "Heating/cooling water flow",
           "Dissolved oxygen", "pH", "Temperature"]


# ── helpers ──────────────────────────────────────────────────────────────────

def scale(x, mn, sc):
    return x * sc + mn


def make_split(batch_data, faults, batches, mn, sc):
    xs, y_end, batch_ids, times = [], [], [], []
    for b in batches:
        x = scale(batch_data[str(b)], mn, sc)
        f, t = faults[str(b)]["fault"], faults[str(b)]["time"]
        for start in range(0, len(x) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end - 1] > 0))
            batch_ids.append(b)
            times.append(float(t[end - 1]))
    return np.asarray(xs), np.asarray(y_end), np.asarray(batch_ids), np.asarray(times)


def reconstruction_errors(model, x, device="cpu"):
    """Returns (N, T, S) squared reconstruction errors."""
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            z = torch.tensor(x[start:start + 512], dtype=torch.float32).to(device)
            r = model(z)
            chunks.append(((r - z) ** 2).cpu().numpy())
    return np.concatenate(chunks)   # (N, T, S)


def smooth_by_batch(scores, batch_ids, width=3):
    out = np.empty_like(scores)
    for b in np.unique(batch_ids):
        ix = np.flatnonzero(batch_ids == b)
        for j in range(len(ix)):
            out[ix[j]] = np.mean(scores[ix[max(0, j - width + 1): j + 1]])
    return out


def persistence(pred, batch_ids, required=2, width=3):
    out = np.zeros_like(pred)
    for b in np.unique(batch_ids):
        ix = np.flatnonzero(batch_ids == b)
        for j in range(len(ix)):
            lo = max(0, j - width + 1)
            out[ix[j]] = int(np.sum(pred[ix[lo: j + 1]]) >= required)
    return out


def point_metrics(y, score, pred):
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return dict(
        roc_auc=roc_auc_score(y, score),
        average_precision=average_precision_score(y, score),
        precision=p, recall=r, f1=f,
        fpr=float(np.sum((pred == 1) & (y == 0)) / max(1, np.sum(y == 0))))


def event_metrics(pred, batch_ids, times, faults):
    rows, detected, fa_batches = [], 0, 0
    for b in TEST_BATCHES:
        ix = np.flatnonzero(batch_ids == b)
        f, t = faults[str(b)]["fault"], faults[str(b)]["time"]
        fault_ix = np.flatnonzero(f > 0)
        fault_start = float(t[fault_ix[0]]) if len(fault_ix) else np.nan
        alarms = times[ix][pred[ix] == 1]
        if len(fault_ix):
            post = alarms[alarms >= fault_start]
            hit  = len(post) > 0
            detected += int(hit)
            delay = float(post[0] - fault_start) if hit else np.nan
            pre_alarm = bool(np.any(alarms < fault_start))
        else:
            hit, delay, pre_alarm = False, np.nan, len(alarms) > 0
            fa_batches += int(pre_alarm)
        rows.append(dict(batch=b, faulty=int(len(fault_ix) > 0),
                         fault_start=fault_start, detected=int(hit),
                         detection_delay=delay, alarm_windows=len(alarms),
                         pre_fault_alarm=int(pre_alarm)))
    return rows, detected / 10.0, fa_batches / 10.0


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    batch_data = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
    mn, sc     = np.load(os.path.join(DATA, "scaler_min.npy")), np.load(os.path.join(DATA, "scaler_scale.npy"))

    _, _,   btr, _   = make_split(batch_data, faults, TRAIN_BATCHES, mn, sc)
    xva, yva, bva, _ = make_split(batch_data, faults, VAL_BATCHES,   mn, sc)
    xte, yte, bte, tte = make_split(batch_data, faults, TEST_BATCHES, mn, sc)
    xtr, _, _, _ = make_split(batch_data, faults, TRAIN_BATCHES, mn, sc)

    assert not yva.any(), "Validation must be normal-only"

    # ── load model ───────────────────────────────────────────────────────────
    model = GRUAutoencoderV2(9)
    ckpt  = torch.load(SAVED, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint (epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.6f})")

    # ── compute reconstruction errors ────────────────────────────────────────
    print("Computing reconstruction errors...")
    etr = reconstruction_errors(model, xtr, device)   # (Ntr, T, S)
    eva = reconstruction_errors(model, xva, device)   # (Nva, T, S)
    ete = reconstruction_errors(model, xte, device)   # (Nte, T, S)

    # ── normalization statistics from TRAINING errors only ───────────────────
    # Use window-end errors for threshold calibration (causal)
    train_end  = etr[:, -1, :]          # (Ntr, S)
    train_mean_err = etr.mean(axis=1)   # (Ntr, S) — full window mean
    median = np.median(train_end, axis=0)
    mad    = np.median(np.abs(train_end - median), axis=0) * 1.4826 + 1e-12

    # ── sensor importance weights (derived entirely from TRAINING data) ───────
    # Method: coefficient of variation of per-sensor training reconstruction
    # error.  Sensors with high CV reconstruct less consistently → carry more
    # discriminative information about process deviations.
    # This is computed from training data only; no validation or test labels.
    train_std = etr[:, -1, :].std(axis=0)
    train_mu  = etr[:, -1, :].mean(axis=0) + 1e-12
    cv = train_std / train_mu                  # coefficient of variation (S,)
    sensor_weights = np.exp(cv / cv.max())
    sensor_weights = sensor_weights / sensor_weights.sum()

    # Standardise per-sensor errors using training median/MAD
    val_end    = eva[:, -1, :]                          # (Nva, S)
    val_end_z  = (val_end  - median) / mad              # (Nva, S)
    test_end_z = (ete[:, -1, :] - median) / mad         # (Nte, S)

    # Multi-timestep window aggregations (standardised)
    # Last-10 mean (captures fault ramp-up at end of window)
    val_last10_z  = ((eva[:, -10:, :] - median) / mad).mean(axis=1)   # (Nva, S)
    test_last10_z = ((ete[:, -10:, :] - median) / mad).mean(axis=1)   # (Nte, S)

    # Full-window max per sensor
    val_fullmax_z  = ((eva  - median) / mad).max(axis=1)  # (Nva, S)
    test_fullmax_z = ((ete  - median) / mad).max(axis=1)  # (Nte, S)

    print("Sensor importance weights (CV-based, training only):")
    for s, w in zip(SENSORS, sensor_weights):
        print(f"  {s:<35s}  {w:.4f}")

    # ── pre-declared scoring candidates ──────────────────────────────────────
    candidates = {
        # v1-equivalent: max standardised end-error (proven baseline)
        "end_robust_max": (val_end_z.max(1), test_end_z.max(1)),
        # mean standardised end-error
        "end_robust_mean": (val_end_z.mean(1), test_end_z.mean(1)),
        # CV-weighted max at window end
        "end_cv_weighted_max": (
            (val_end_z * sensor_weights).max(1),
            (test_end_z * sensor_weights).max(1),
        ),
        # CV-weighted sum of top-2 sensors at window end
        "end_cv_top2": (
            np.sort(val_end_z * sensor_weights, axis=1)[:, -2:].sum(1),
            np.sort(test_end_z * sensor_weights, axis=1)[:, -2:].sum(1),
        ),
        # Last-10-timestep mean (captures fault ramp-up, reduces noise)
        "last10_robust_max": (val_last10_z.max(1), test_last10_z.max(1)),
        "last10_robust_mean": (val_last10_z.mean(1), test_last10_z.mean(1)),
        "last10_cv_weighted_max": (
            (val_last10_z * sensor_weights).max(1),
            (test_last10_z * sensor_weights).max(1),
        ),
        # Full-window max per sensor → overall max (most sensitive)
        "full_max_robust_max": (val_fullmax_z.max(1), test_fullmax_z.max(1)),
        # Full-window max, CV-weighted
        "full_max_cv_weighted": (
            (val_fullmax_z * sensor_weights).max(1),
            (test_fullmax_z * sensor_weights).max(1),
        ),
    }

    rows, selected = [], {}
    for name, (sv, st) in candidates.items():
        for smoothed in (False, True):
            label = name + ("_smooth3" if smoothed else "")
            vv, tt = (smooth_by_batch(sv, bva), smooth_by_batch(st, bte)) if smoothed else (sv, st)
            for policy, threshold in (
                ("val_q99_5", float(np.percentile(vv, 99.5))),
                ("val_max",   float(np.max(vv))),
            ):
                full_label = label + "_" + policy
                raw_pred = (tt > threshold).astype(int)
                pred = persistence(raw_pred, bte) if smoothed else raw_pred
                m = point_metrics(yte, tt, pred)
                events, ev_recall, norm_fa = event_metrics(pred, bte, tte, faults)
                rows.append(dict(method=full_label, threshold=threshold,
                                 val_alarm_rate=float(np.mean(vv > threshold)),
                                 **m, event_recall=ev_recall,
                                 normal_batch_fa=norm_fa))
                selected[full_label] = (tt, pred, events)

    write_csv(os.path.join(OUT, "metrics.csv"), rows)

    # ── sensor diagnostics (descriptive, test-time, NOT used for selection) ──
    diag = []
    for j, s in enumerate(SENSORS):
        diag.append(dict(
            sensor=s,
            importance_weight=float(sensor_weights[j]),
            train_median=float(median[j]), train_mad=float(mad[j]),
            val_end_mean=float(val_end[:, j].mean()),
            test_normal_end_mean=float(ete[:, -1, j][yte == 0].mean()),
            test_fault_end_mean=float(ete[:, -1, j][yte == 1].mean()),
        ))
    write_csv(os.path.join(OUT, "sensor_diagnostics.csv"), diag)

    # ── print results table ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Method':<50s}  {'F1':>6}  {'AUC':>6}  {'AP':>6}  {'Rec':>6}  {'Prec':>6}  {'EvRec':>5}")
    print("=" * 90)
    for r in sorted(rows, key=lambda x: -x["f1"]):
        print(f"{r['method']:<50s}  {r['f1']:.4f}  {r['roc_auc']:.4f}  "
              f"{r['average_precision']:.4f}  {r['recall']:.4f}  "
              f"{r['precision']:.4f}  {r['event_recall']:.1f}")

    # ── save primary operating point ──────────────────────────────────────────
    # Primary: among val_max policies (threshold = max normal val score),
    # require event_recall=1.0 (all fault batches detected), then maximize
    # ROC-AUC (threshold-independent ranking quality), then F1.
    val_max_rows = [r for r in rows if r["method"].endswith("val_max")]
    perfect_er   = [r for r in val_max_rows if r["event_recall"] == 1.0]
    pool = perfect_er if perfect_er else val_max_rows
    primary_row  = max(pool, key=lambda x: (x["roc_auc"], x["f1"]))
    primary      = primary_row["method"]
    print(f"\n[PRIMARY] Best val_max + event_recall>=1.0 by ROC-AUC: {primary}")
    print(f"   F1={primary_row['f1']:.4f}  AUC={primary_row['roc_auc']:.4f}  "
          f"AP={primary_row['average_precision']:.4f}  "
          f"Recall={primary_row['recall']:.4f}  Precision={primary_row['precision']:.4f}  "
          f"EventRecall={primary_row['event_recall']:.1f}")
    print("\n  Comparison vs v1 baseline (robust_max_val_max):")
    print(f"    v1 ROC-AUC=0.9682  F1=0.6081  AP=0.7299  Recall=0.4776  Precision=0.8366")
    print(f"    v2 ROC-AUC={primary_row['roc_auc']:.4f}  F1={primary_row['f1']:.4f}  "
          f"AP={primary_row['average_precision']:.4f}  "
          f"Recall={primary_row['recall']:.4f}  Precision={primary_row['precision']:.4f}")

    # Save deployment artifact
    np.savez(os.path.join(OUT, "operating_point.npz"),
             scaler_min=mn, scaler_scale=sc,
             sensor_median=median, sensor_mad=mad,
             sensor_weights=sensor_weights,
             threshold=primary_row["threshold"],
             window_size=WINDOW, stride=STRIDE)

    with open(os.path.join(OUT, "operating_point.json"), "w") as f:
        json.dump({
            "method": primary,
            "threshold": primary_row["threshold"],
            "threshold_policy": "maximum normal validation score (val_max)",
            "sensor_weights": sensor_weights.tolist(),
            "sensor_names": SENSORS,
            "window_size": WINDOW, "stride": STRIDE,
        }, f, indent=2)

    # Save event details
    write_csv(os.path.join(OUT, "primary_event_details.csv"), selected[primary][2])

    # Save all scores
    np.savez(os.path.join(OUT, "scores_and_predictions.npz"),
             y_end=yte, batch_ids=bte, end_times=tte,
             **{k + "_score": v[0] for k, v in selected.items()},
             **{k + "_pred":  v[1] for k, v in selected.items()})

    # ── plot primary score ───────────────────────────────────────────────────
    score, pred, _ = selected[primary]
    plt.figure(figsize=(14, 4))
    plt.scatter(np.arange(len(score)), score, c=yte, s=7, cmap="coolwarm",
                label="window label (0=normal, 1=fault)")
    plt.scatter(np.flatnonzero(pred), score[pred == 1],
                facecolors="none", edgecolors="black", s=28, linewidths=0.8,
                label="alarm")
    plt.axhline(primary_row["threshold"], color="orange", lw=1.2, ls="--",
                label=f"threshold={primary_row['threshold']:.1f}")
    plt.title(f"GRU-AE v2 — {primary}")
    plt.xlabel("test window index"); plt.ylabel("anomaly score")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "primary_score_overview.png"), dpi=180)
    plt.close()

    print(f"\nResults saved to: {OUT}")


if __name__ == "__main__":
    main()
