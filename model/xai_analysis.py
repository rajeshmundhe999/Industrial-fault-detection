"""
SHAP-based Fault Isolation for GRU-AE v2
==========================================
Uses Kernel SHAP on the flattened (60-timestep × 9-sensor = 540 feature)
reconstruction error to explain which sensors contribute most to each anomaly
score for detected faulty windows.

Method
------
1. Load the trained GRU-AE v2 and the leakage-safe operating point.
2. For each detected-anomalous test window, compute the anomaly score
   (importance-weighted max standardised reconstruction error).
3. Run SHAP KernelExplainer using a random subsample of NORMAL training
   windows as the background distribution.  This is fully valid: background
   is train-normal, explained instances are test anomalies.
4. Aggregate SHAP values by sensor (sum absolute SHAP across all timesteps
   for each sensor) to get a per-sensor fault contribution ranking.
5. Save per-window sensor rankings, summary plots, and a JSON report.

Output
------
model/results/xai/
  shap_sensor_ranks.csv       — per-anomalous-window sensor ranking table
  shap_summary_barplot.png    — average absolute SHAP by sensor (bar chart)
  shap_beeswarm_top5.png      — beeswarm for top-5 sensors
  fault_isolation_report.json — JSON report for dashboard integration
"""
import json, os, sys, time
import numpy as np
import torch
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
OP    = os.path.join(BASE, "model", "results", "leakage_safe_v2", "operating_point.npz")
SAVED = os.path.join(BASE, "model", "saved", "gru_ae_v2_best.pt")
OUT   = os.path.join(BASE, "model", "results", "xai")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

SENSORS = ["Agitator RPM", "Aeration rate", "Sugar feed rate",
           "Acid flow rate", "Base flow rate", "Heating/cooling water flow",
           "Dissolved oxygen", "pH", "Temperature"]
N_BACKGROUND  = 200   # number of normal training windows for SHAP background
N_EXPLAIN_MAX = 50    # max anomalous windows to explain (speed)


def load_model(path, device="cpu"):
    model = GRUAutoencoderV2(9)
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def scale(x, mn, sc):
    return x * sc + mn


def make_test_split(batch_data, faults, mn, sc, window=60, stride=5):
    xs, y_end, batch_ids, times = [], [], [], []
    for b in range(81, 101):
        x = scale(batch_data[str(b)], mn, sc)
        f, t = faults[str(b)]["fault"], faults[str(b)]["time"]
        for start in range(0, len(x) - window + 1, stride):
            end = start + window
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end - 1] > 0))
            batch_ids.append(b)
            times.append(float(t[end - 1]))
    return np.asarray(xs), np.asarray(y_end), np.asarray(batch_ids), np.asarray(times)


def anomaly_score_fn(windows_flat, model, op, device):
    """
    windows_flat: (N, 60*9) numpy array of flattened raw-scale windows
    Returns (N,) anomaly scores.
    """
    N = windows_flat.shape[0]
    windows = windows_flat.reshape(N, 60, 9).astype(np.float32)
    t = torch.tensor(windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(t)
        err   = ((recon - t) ** 2).cpu().numpy()   # (N, 60, 9)

    # Standardise per-sensor using training stats
    median  = op["sensor_median"]
    mad     = op["sensor_mad"]
    weights = op["sensor_weights"]

    end_err_z = (err[:, -1, :] - median) / mad     # (N, 9)
    # importance-weighted max
    scores = (end_err_z * weights).max(axis=1)
    return scores


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ── load data ────────────────────────────────────────────────────────────
    batch_data = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
    op         = np.load(OP)
    mn, sc_arr = op["scaler_min"], op["scaler_scale"]

    X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)  # (N, 60, 9) scaled
    xte, yte, bte, tte = make_test_split(batch_data, faults, mn, sc_arr)

    model = load_model(SAVED, device)
    print(f"Model loaded. Test windows: {len(xte)}, fault-labelled: {int(yte.sum())}")

    # ── compute all reconstruction errors ─────────────────────────────────────
    def get_end_errors(windows, mdl, dev):
        """Returns (N, 9) per-sensor squared recon error at window end."""
        chunks = []
        mdl.eval()
        with torch.no_grad():
            for s in range(0, len(windows), 256):
                z = torch.tensor(windows[s:s+256], dtype=torch.float32).to(dev)
                r = mdl(z)
                chunks.append(((r - z) ** 2)[:, -1, :].cpu().numpy())
        return np.concatenate(chunks)  # (N, 9)

    # Training errors for normalization
    X_train_raw = X_train  # already in raw sensor scale (same as xte)
    err_train = get_end_errors(X_train_raw, model, device)  # (Ntr, 9)
    err_test  = get_end_errors(xte,         model, device)  # (Nte, 9)

    median  = np.median(err_train, axis=0)
    mad_arr = np.median(np.abs(err_train - median), axis=0) * 1.4826 + 1e-12

    # Standardised errors
    err_test_z = (err_test - median) / mad_arr   # (Nte, 9)

    # ── load operating point for threshold ────────────────────────────────────
    op        = np.load(OP)
    threshold = float(op["threshold"])
    sw        = op["sensor_weights"] if "sensor_weights" in op else np.ones(9) / 9

    scores = (err_test_z * sw).max(axis=1)
    pred   = (scores > threshold).astype(int)
    tp_ix  = np.flatnonzero((pred == 1) & (yte == 1))
    fa_ix  = np.flatnonzero((pred == 1) & (yte == 0))
    print(f"True positives (TP) to explain: {len(tp_ix)}  False alarms: {len(fa_ix)}")

    # ── SHAP in sensor-error space (9-dim, fast and interpretable) ────────────
    # Score function: takes (N, 9) standardised sensor errors → (N,) score
    # Background: standardised training errors (normal operation)
    err_train_z = (err_train - median) / mad_arr  # (Ntr, 9)

    rng    = np.random.default_rng(42)
    bg_idx = rng.choice(len(err_train_z), size=min(N_BACKGROUND, len(err_train_z)), replace=False)
    background = err_train_z[bg_idx]     # (N_BG, 9) — normal sensor errors

    def score_sensor_errors(se):
        """se: (N, 9) standardised sensor errors → (N,) anomaly score."""
        return (se * sw).max(axis=1)

    explain_ix  = tp_ix[:N_EXPLAIN_MAX]
    explain_se  = err_test_z[explain_ix]   # (N_explain, 9)

    print(f"Running SHAP KernelExplainer: {len(explain_ix)} TP windows, "
          f"{len(background)} background samples (9-dim sensor space)...")
    t0 = time.time()
    explainer   = shap.KernelExplainer(score_sensor_errors, background)
    shap_values = explainer.shap_values(explain_se, nsamples=512)
    sv = np.asarray(shap_values)  # (N_explain, 9)
    print(f"SHAP done in {time.time()-t0:.1f}s")

    # ── aggregate SHAP by sensor ──────────────────────────────────────────────
    # sv shape: (N_explain, 9) — direct per-sensor attribution
    sv_abs    = np.abs(sv)             # (N_explain, 9)
    mean_sv   = sv_abs.mean(axis=0)   # (9,)
    rank_idx  = np.argsort(mean_sv)[::-1]

    print("\nSensor fault contribution ranking (mean |SHAP| across anomalous windows):")
    for rank, j in enumerate(rank_idx):
        print(f"  {rank+1}. {SENSORS[j]:<35s}  mean|SHAP|={mean_sv[j]:.6f}")

    # ── save per-window ranking CSV ───────────────────────────────────────────
    import csv
    rows = []
    for i, w_idx in enumerate(explain_ix):
        rank = np.argsort(sv_abs[i])[::-1]
        row  = {"window_idx": int(w_idx), "batch": int(bte[w_idx]),
                "time": float(tte[w_idx]), "label": int(yte[w_idx]),
                "score": float(scores[w_idx])}
        for r, j in enumerate(rank):
            row[f"rank{r+1}_sensor"] = SENSORS[j]
            row[f"rank{r+1}_shap"]   = float(sv_abs[i, j])
        rows.append(row)

    with open(os.path.join(OUT, "shap_sensor_ranks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ── save JSON report ──────────────────────────────────────────────────────
    report = {
        "n_anomalous_windows_explained": int(len(explain_ix)),
        "n_background_windows": int(len(background)),
        "shap_space": "9-dim standardised sensor reconstruction error at window end",
        "sensor_ranking": [
            {"rank": int(r+1), "sensor": SENSORS[j],
             "mean_abs_shap": float(mean_sv[j])}
            for r, j in enumerate(rank_idx)
        ],
        "interpretation": (
            "Higher mean |SHAP| value indicates that sensor's reconstruction error "
            "contributed more to the anomaly score across detected fault windows. "
            "This is descriptive attribution — it shows WHICH sensors are anomalous, "
            "not necessarily the physical root cause of the fault."
        )
    }
    with open(os.path.join(OUT, "fault_isolation_report.json"), "w") as f:
        json.dump(report, f, indent=2)


    # ── bar plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#e74c3c" if j == rank_idx[0] else "#3498db" for j in range(9)]
    sorted_sensors = [SENSORS[j] for j in rank_idx]
    sorted_vals    = mean_sv[rank_idx]
    ax.barh(sorted_sensors[::-1], sorted_vals[::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Mean |SHAP| (contribution to anomaly score)", fontsize=11)
    ax.set_title("Sensor Fault Contribution — SHAP Attribution\n(GRU-AE v2, normal-only training)", fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_summary_barplot.png"), dpi=180)
    plt.close()

    # ── SHAP beeswarm-style scatter (top 5 sensors) ───────────────────────────
    top5 = rank_idx[:5]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)
    for ax_i, j in enumerate(top5):
        ax = axes[ax_i]
        vals = sv_abs[:, j]
        jitter = rng.uniform(-0.3, 0.3, len(vals))
        sc = ax.scatter(vals, jitter, c=scores[explain_ix], cmap="Reds",
                        s=40, alpha=0.8, edgecolors="none")
        ax.set_title(SENSORS[j].replace(" ", "\n"), fontsize=9)
        ax.set_xlabel("|SHAP|", fontsize=8)
        ax.set_yticks([])
        ax.spines[["top","right","left"]].set_visible(False)
    plt.suptitle("Per-sensor |SHAP| for detected anomalous windows", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_beeswarm_top5.png"), dpi=180)
    plt.close()

    print(f"\nXAI outputs saved to: {OUT}")
    print("  shap_sensor_ranks.csv")
    print("  fault_isolation_report.json")
    print("  shap_summary_barplot.png")
    print("  shap_beeswarm_top5.png")


if __name__ == "__main__":
    main()
