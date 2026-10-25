"""
Monte Carlo Dropout Uncertainty Quantification for GRU-AE v2
=============================================================
Produces calibrated confidence intervals on the anomaly score.

Method
------
The GRU-AE v2 has Dropout(0.15) layers. Normally these are disabled at
inference (model.eval()). By keeping them ACTIVE during inference and
running N stochastic forward passes, we get a distribution over anomaly
scores for each window. This is Monte Carlo Dropout (Gal & Ghahramani 2016).

Outputs
-------
- mean_score:   point estimate (mean of N passes)
- std_score:    epistemic uncertainty (std of N passes)
- alarm:        mean_score > threshold  (same threshold as deterministic eval)
- confident:    std_score < val_std_95th_percentile (alarm is not noisy)
- For each test window: mean, std, 5th/95th percentile of score distribution

Saved to model/results/uncertainty/
"""
import csv, json, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
OP    = os.path.join(BASE, "model", "results", "leakage_safe_v2", "operating_point.npz")
SAVED = os.path.join(BASE, "model", "saved", "gru_ae_v2_best.pt")
OUT   = os.path.join(BASE, "model", "results", "uncertainty")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

N_MC      = 50    # Monte Carlo samples
WINDOW    = 60
STRIDE    = 5
SENSORS   = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
             "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]


def scale(x, mn, sc): return x * sc + mn

def make_split(batch_data, faults, batches, mn, sc):
    xs, y_end, batch_ids, times = [], [], [], []
    for b in batches:
        x = scale(batch_data[str(b)], mn, sc)
        f, t = faults[str(b)]["fault"], faults[str(b)]["time"]
        for start in range(0, len(x) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end-1] > 0))
            batch_ids.append(b); times.append(float(t[end-1]))
    return np.asarray(xs), np.asarray(y_end), np.asarray(batch_ids), np.asarray(times)


def mc_scores(model, x, median, mad, weights, n_mc=N_MC, batch_size=256):
    """Run N_MC stochastic forward passes, return (N, n_mc) score matrix."""
    model.train()  # keep dropout ON — this is intentional for MC-Dropout
    all_scores = []
    with torch.no_grad():
        for _ in range(n_mc):
            run = []
            for s in range(0, len(x), batch_size):
                z = torch.tensor(x[s:s+batch_size], dtype=torch.float32)
                r = model(z)
                err = ((r - z)**2)[:, -1, :].numpy()
                err_z = (err - median) / mad
                run.append((err_z * weights).max(axis=1))
            all_scores.append(np.concatenate(run))
    return np.stack(all_scores, axis=1)  # (N, n_mc)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("MC-Dropout Uncertainty Quantification")
    print(f"N_MC = {N_MC} stochastic forward passes")

    batch_data = np.load(os.path.join(DATA,"batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA,"batch_faults.npy"), allow_pickle=True).item()
    op         = np.load(OP)
    mn, sc_arr = op["scaler_min"], op["scaler_scale"]
    median, mad = op["sensor_median"], op["sensor_mad"]
    weights     = op["sensor_weights"]
    threshold   = float(op["threshold"])

    xva, yva, bva, _   = make_split(batch_data, faults, range(71,81), mn, sc_arr)
    xte, yte, bte, tte = make_split(batch_data, faults, range(81,101), mn, sc_arr)

    model = GRUAutoencoderV2(9)
    ckpt  = torch.load(SAVED, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Calibrate uncertainty threshold from NORMAL VALIDATION windows
    print("Computing validation MC scores for uncertainty calibration...")
    val_mc  = mc_scores(model, xva, median, mad, weights)   # (Nva, n_mc)
    val_std = val_mc.std(axis=1)                            # (Nva,)
    # 95th percentile std on normal windows → above this = high uncertainty
    unc_threshold = float(np.percentile(val_std, 95))
    print(f"Uncertainty threshold (val std 95th pct): {unc_threshold:.4f}")

    # Test MC scores
    print("Computing test MC scores...")
    test_mc   = mc_scores(model, xte, median, mad, weights)  # (Nte, n_mc)
    mean_score = test_mc.mean(axis=1)
    std_score  = test_mc.std(axis=1)
    p5_score   = np.percentile(test_mc, 5, axis=1)
    p95_score  = np.percentile(test_mc, 95, axis=1)

    # Alarms
    alarm      = (mean_score > threshold).astype(int)
    confident  = (std_score  < unc_threshold).astype(int)
    # High-confidence alarm: mean exceeds threshold AND uncertainty is low
    hc_alarm   = ((mean_score > threshold) & (std_score < unc_threshold)).astype(int)

    def metrics(y, pred, score):
        if pred.sum() == 0: return dict(f1=0,precision=0,recall=0,roc_auc=roc_auc_score(y,score),ap=average_precision_score(y,score),fpr=0)
        p,r,f,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        fpr = float(np.sum((pred==1)&(y==0)) / max(1,np.sum(y==0)))
        return dict(f1=f, precision=p, recall=r,
                    roc_auc=roc_auc_score(y,score),
                    ap=average_precision_score(y,score), fpr=fpr)

    m_std  = metrics(yte, alarm,    mean_score)
    m_hc   = metrics(yte, hc_alarm, mean_score)

    print("\n--- Standard MC-mean alarm (same threshold as deterministic) ---")
    print(f"  ROC-AUC={m_std['roc_auc']:.4f}  F1={m_std['f1']:.4f}  "
          f"Recall={m_std['recall']:.4f}  Precision={m_std['precision']:.4f}  FPR={m_std['fpr']:.4f}")
    print("\n--- High-confidence alarm (mean>threshold AND std<unc_threshold) ---")
    print(f"  ROC-AUC={m_hc['roc_auc']:.4f}  F1={m_hc['f1']:.4f}  "
          f"Recall={m_hc['recall']:.4f}  Precision={m_hc['precision']:.4f}  FPR={m_hc['fpr']:.4f}")
    print(f"\n  High-confidence alarms: {hc_alarm.sum()} / {alarm.sum()} alarms "
          f"({100*hc_alarm.sum()/max(1,alarm.sum()):.1f}% of alarms are confident)")

    # Save per-window CSV
    rows = []
    for i in range(len(yte)):
        rows.append(dict(window=i, batch=int(bte[i]), time=float(tte[i]),
                         label=int(yte[i]), mean_score=float(mean_score[i]),
                         std_score=float(std_score[i]), p5=float(p5_score[i]),
                         p95=float(p95_score[i]), alarm=int(alarm[i]),
                         confident=int(confident[i]), hc_alarm=int(hc_alarm[i])))
    with open(os.path.join(OUT,"mc_dropout_scores.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Save summary JSON
    summary = dict(n_mc=N_MC, unc_threshold=unc_threshold, threshold=threshold,
                   mc_alarm=m_std, hc_alarm=m_hc,
                   pct_alarms_confident=float(100*hc_alarm.sum()/max(1,alarm.sum())))
    with open(os.path.join(OUT,"uncertainty_summary.json"),"w") as f:
        json.dump(summary, f, indent=2)

    # Plot: anomaly score with uncertainty bands, first 4 test batches
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    test_batches = list(range(81, 85))
    for ax, b in zip(axes.flat, test_batches):
        ix = np.flatnonzero(bte == b)
        t  = tte[ix]; mu = mean_score[ix]; sd = std_score[ix]
        f_info = faults[str(b)]
        fault_t = float(f_info["time"][np.flatnonzero(f_info["fault"]>0)[0]]) if f_info["fault"].any() else None
        ax.fill_between(t, mu-2*sd, mu+2*sd, alpha=0.25, color="steelblue", label="±2σ")
        ax.plot(t, mu, color="steelblue", lw=1.2, label="mean score")
        ax.axhline(threshold, color="red", ls="--", lw=1, label="threshold")
        if fault_t: ax.axvline(fault_t, color="orange", ls=":", lw=1.5, label="fault onset")
        ax.set_title(f"Batch {b}"); ax.set_xlabel("time (h)"); ax.set_ylabel("score")
        ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)
    plt.suptitle("MC-Dropout Anomaly Score with Uncertainty Bands (±2σ)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"mc_uncertainty_bands.png"), dpi=180)
    plt.close()

    # Plot: uncertainty vs score scatter (shows separation of confident alarms)
    fig, ax = plt.subplots(figsize=(8,5))
    colors = np.where(yte==1, "red", "steelblue")
    ax.scatter(mean_score, std_score, c=colors, s=5, alpha=0.4)
    ax.axvline(threshold, color="red", ls="--", lw=1, label=f"score threshold={threshold:.1f}")
    ax.axhline(unc_threshold, color="orange", ls="--", lw=1,
               label=f"uncertainty threshold={unc_threshold:.4f}")
    ax.set_xlabel("Mean anomaly score"); ax.set_ylabel("Std (uncertainty)")
    ax.set_title("Score vs Uncertainty — MC-Dropout\n(red=fault, blue=normal)")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"score_vs_uncertainty.png"), dpi=180)
    plt.close()

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
