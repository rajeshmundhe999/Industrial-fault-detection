"""
Early Fault Prognosis — Alarm Lead Time Analysis
=================================================
Computes how many timesteps BEFORE a fault becomes detectable the GRU-AE
score already shows a significant rising trend.

Method
------
1. Load per-window test scores (already computed in leakage_safe_v2).
2. For each test batch reconstruct a time-indexed score curve.
3. On normal batches (81-90), fit an exponential smoothing trend of the
   score curve → derive a "trend alarm threshold" from the 99.5th percentile
   of the smoothed derivative on normal data.
4. On faulty batches (91-100), find:
   a) Static alarm timestep ts_static  (score > main_threshold)
   b) Trend alarm  timestep ts_trend   (|score_EMA_rate| > trend_threshold)
5. Report LEAD TIME = ts_static - ts_trend  for each fault batch.
   Positive = trend alarm fires BEFORE static alarm (early warning).
6. Also define a COMPOSITE SCORE:
   composite = alpha * score + (1-alpha) * (EMA rate of score)
   which combines current value and rising trend into one signal.
7. Evaluate composite vs static with ROC-AUC comparison.

Outputs
-------
model/results/early_detection/
  lead_time_per_batch.csv        — per-batch lead times
  composite_vs_static_auc.csv   — AUC comparison table
  lead_time_histogram.png        — distribution of lead times
  score_curves_faulty.png        — per-batch score curves with alarm markers
"""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
OP_V2 = os.path.join(BASE, "model", "results", "leakage_safe_v2", "scores_and_predictions.npz")
OUT   = os.path.join(BASE, "model", "results", "early_detection")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BASE)

# Smoothing parameter for exponential moving average
ALPHA_EMA    = 0.3    # EMA smoothing (lower = smoother)
TREND_WINDOW = 5      # derivative window (timesteps in score space)
COMPOSITE_ALPHA = 0.7 # weight on current score vs rate-of-change

SENSORS = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
           "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]


def ema_series(arr, alpha=ALPHA_EMA):
    """Exponential moving average of 1-D array."""
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def score_rate(scores_1d, window=TREND_WINDOW):
    """Rate-of-change of the score curve (forward diff over `window` steps)."""
    rate = np.zeros_like(scores_1d)
    for i in range(window, len(scores_1d)):
        rate[i] = (scores_1d[i] - scores_1d[i - window]) / window
    return rate


def first_alarm(arr, thresh):
    """Index of first element > thresh, or -1."""
    idxs = np.flatnonzero(arr > thresh)
    return int(idxs[0]) if len(idxs) > 0 else -1


def main():
    print("=" * 60)
    print("EARLY FAULT PROGNOSIS ANALYSIS")
    print("=" * 60)

    # ── load saved scores ─────────────────────────────────────────────────────
    sp = np.load(OP_V2)
    # Primary scoring policy: end_robust_mean_val_max (pre-declared, val-only threshold)
    scores    = sp["end_robust_mean_val_max_score"]  # (N_windows,) float
    labels    = sp["y_end"]                           # (N_windows,) int
    batch_ids = sp["batch_ids"]                       # (N_windows,) int
    times     = sp["end_times"] if "end_times" in sp else np.zeros(len(scores))

    # Load main threshold (val_max policy from operating_point.npz)
    op_path = os.path.join(BASE, "model", "results", "leakage_safe_v2", "operating_point.npz")
    op   = np.load(op_path, allow_pickle=True)
    main_threshold = float(op["threshold"])
    print(f"Static threshold (val_max): {main_threshold:.4f}")

    # ── reconstruct per-batch time-indexed score curves ───────────────────────
    batches = sorted(np.unique(batch_ids))
    normal_batches = [b for b in batches if b <= 90]
    faulty_batches = [b for b in batches if b >= 91]
    print(f"Normal test batches: {normal_batches}")
    print(f"Faulty test batches: {faulty_batches}")

    # Compute EMA of score and rate-of-change per batch
    batch_curves = {}
    for b in batches:
        ix     = np.flatnonzero(batch_ids == b)
        sc     = scores[ix]
        lb     = labels[ix]
        t      = times[ix] if np.any(times > 0) else np.arange(len(ix))
        ema_sc = ema_series(sc, ALPHA_EMA)
        rate   = score_rate(ema_sc, TREND_WINDOW)
        batch_curves[b] = dict(sc=sc, lb=lb, t=t, ema=ema_sc, rate=rate)

    # ── derive trend threshold from normal batches ────────────────────────────
    all_normal_rates = np.concatenate([batch_curves[b]["rate"] for b in normal_batches])
    trend_threshold  = np.percentile(all_normal_rates, 99.5)
    print(f"Trend threshold (normal rate 99.5th pct): {trend_threshold:.4f}")

    # ── lead time analysis on faulty batches ─────────────────────────────────
    rows = []
    for b in faulty_batches:
        sc   = batch_curves[b]["sc"]
        ema  = batch_curves[b]["ema"]
        rate = batch_curves[b]["rate"]
        lb   = batch_curves[b]["lb"]

        ts_static = first_alarm(sc,   main_threshold)
        ts_trend  = first_alarm(rate, trend_threshold)
        ts_fault  = first_alarm(lb,   0.5)   # first truly faulty window

        lead_time = (ts_static - ts_trend) if (ts_static > 0 and ts_trend > 0) else None
        early     = (lead_time > 0) if lead_time is not None else False

        print(f"  Batch {b}: fault_at={ts_fault}  trend_alarm={ts_trend}  "
              f"static_alarm={ts_static}  lead={lead_time}  "
              f"{'EARLY' if early else 'LATE/SAME'}")

        rows.append(dict(
            batch=int(b),
            fault_window_idx=int(ts_fault) if ts_fault >= 0 else "none",
            trend_alarm_idx=int(ts_trend) if ts_trend >= 0 else "none",
            static_alarm_idx=int(ts_static) if ts_static >= 0 else "none",
            lead_time_windows=int(lead_time) if lead_time is not None else "none",
            early_warning=bool(early),
        ))

    # ── composite score evaluation ────────────────────────────────────────────
    # Composite = ALPHA*score + (1-ALPHA)*normalized_rate
    # Normalize rate to same scale as score using normal stats
    all_rates = np.concatenate([batch_curves[b]["rate"] for b in batches])
    rate_std  = np.std(all_rates) + 1e-12
    all_sc    = np.concatenate([batch_curves[b]["sc"]   for b in batches])
    composite_scores = (COMPOSITE_ALPHA * all_sc +
                        (1 - COMPOSITE_ALPHA) * (all_rates / rate_std))
    all_labels = np.concatenate([batch_curves[b]["lb"]  for b in batches])

    auc_static    = roc_auc_score(all_labels, all_sc)
    auc_composite = roc_auc_score(all_labels, composite_scores)
    ap_static     = average_precision_score(all_labels, all_sc)
    ap_composite  = average_precision_score(all_labels, composite_scores)

    print(f"\nStatic score    AUC={auc_static:.4f}  AP={ap_static:.4f}")
    print(f"Composite score AUC={auc_composite:.4f}  AP={ap_composite:.4f}")

    # Composite alarm at optimal composite threshold (from normal composite)
    normal_composite = np.concatenate(
        [COMPOSITE_ALPHA * batch_curves[b]["sc"] +
         (1 - COMPOSITE_ALPHA) * (batch_curves[b]["rate"] / rate_std)
         for b in normal_batches])
    comp_threshold = float(np.percentile(normal_composite, 99.5))
    pred_static    = (all_sc > main_threshold).astype(int)
    pred_composite = (composite_scores > comp_threshold).astype(int)

    p_s, r_s, f1_s, _ = precision_recall_fscore_support(
        all_labels, pred_static, average="binary", zero_division=0)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        all_labels, pred_composite, average="binary", zero_division=0)

    print(f"\nStatic    F1={f1_s:.4f}  Recall={r_s:.4f}  Precision={p_s:.4f}")
    print(f"Composite F1={f1_c:.4f}  Recall={r_c:.4f}  Precision={p_c:.4f}")

    # ── save results ──────────────────────────────────────────────────────────
    with open(os.path.join(OUT, "lead_time_per_batch.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(OUT, "composite_vs_static_auc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "roc_auc", "avg_precision", "f1", "recall", "precision"])
        w.writerow(["static",    round(auc_static, 4),    round(ap_static, 4),
                    round(f1_s, 4), round(r_s, 4), round(p_s, 4)])
        w.writerow(["composite", round(auc_composite, 4), round(ap_composite, 4),
                    round(f1_c, 4), round(r_c, 4), round(p_c, 4)])

    # ── lead time histogram ───────────────────────────────────────────────────
    lead_vals = [r["lead_time_windows"] for r in rows
                 if isinstance(r["lead_time_windows"], int)]
    fig, ax = plt.subplots(figsize=(8, 4))
    if lead_vals:
        ax.bar(range(len(lead_vals)), lead_vals,
               color=["forestgreen" if v > 0 else "tomato" for v in lead_vals])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(lead_vals)))
        ax.set_xticklabels([f"B{r['batch']}" for r in rows
                            if isinstance(r["lead_time_windows"], int)], rotation=45)
        ax.set_ylabel("Lead time (windows; >0 = early warning)")
        ax.set_title("Early Detection Lead Time per Fault Batch\n"
                     "(green=trend alarm before static alarm, red=late/same)")
    else:
        ax.text(0.5, 0.5, "No lead time data available", ha="center", va="center",
                transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lead_time_histogram.png"), dpi=180)
    plt.close()

    # ── per-batch score curves ────────────────────────────────────────────────
    ncols = 2
    nrows = (len(faulty_batches) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, b in enumerate(faulty_batches):
        ax   = axes_flat[i]
        sc   = batch_curves[b]["sc"]
        ema  = batch_curves[b]["ema"]
        rate = batch_curves[b]["rate"] / (rate_std + 1e-12)
        lb   = batch_curves[b]["lb"]
        xs   = np.arange(len(sc))
        ax.fill_between(xs, 0, lb * sc.max() * 1.1,
                        alpha=0.15, color="red", label="fault region")
        ax.plot(xs, sc,  color="steelblue", linewidth=1.0, label="score")
        ax.plot(xs, ema, color="navy", linewidth=1.5, linestyle="--", label="EMA")
        ax.axhline(main_threshold, color="tomato",    linewidth=1.2,
                   linestyle="-",  label="static thr")
        ax.set_title(f"Batch {b}", fontsize=9)
        ax.set_xlabel("Window idx"); ax.set_ylabel("Score")
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.spines[["top","right"]].set_visible(False)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("GRU-AE v2 Score Curves — Faulty Batches\n"
                 "(blue=raw score, navy--=EMA, red thr=static alarm)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "score_curves_faulty.png"), dpi=180)
    plt.close()

    print(f"\nSaved to {OUT}")
    print("  lead_time_per_batch.csv")
    print("  composite_vs_static_auc.csv")
    print("  lead_time_histogram.png")
    print("  score_curves_faulty.png")


if __name__ == "__main__":
    main()
