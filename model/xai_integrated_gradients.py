"""
Integrated Gradients — Scientific XAI for GRU-AE v2
=====================================================
Unlike SHAP on a hand-crafted score formula, Integrated Gradients (IG)
attributes the anomaly score DIRECTLY to the input sensor-timestep values
by integrating gradients through the actual GRU network.

Method (Sundararajan et al. 2017, "Axiomatic Attribution for Deep Networks")
---------------------------------------------------------------------------
IG_i(x) = (x_i - baseline_i) × ∫₀¹ ∂F(baseline + α(x-baseline))/∂x_i dα

where:
  x        = input window (60×9)
  baseline = mean normal training window (expected process state)
  F(x)     = anomaly score = max over sensors of standardised reconstruction error
  integral approximated by 50 Riemann steps

This attributes the anomaly score to EACH (timestep, sensor) input feature.
We then aggregate over timesteps to get per-sensor attribution, and over
sensors to get per-timestep attribution (temporal fault onset).

Outputs
-------
model/results/integrated_gradients/
  ig_sensor_attributions.csv      — per-window sensor-level IG attribution
  ig_temporal_attributions.csv    — per-window timestep-level attribution
  ig_summary_report.json          — aggregate ranking
  ig_sensor_barplot.png           — mean |IG| by sensor across all faults
  ig_temporal_heatmap.png         — temporal attribution heatmap (fault windows)
  ig_vs_shap_comparison.png       — IG vs SHAP sensor ranking comparison
"""
import csv, json, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
OP    = os.path.join(BASE, "model", "results", "leakage_safe_v2", "operating_point.npz")
SAVED = os.path.join(BASE, "model", "saved", "gru_ae_v2_best.pt")
OUT   = os.path.join(BASE, "model", "results", "integrated_gradients")
SHAP_JSON = os.path.join(BASE, "model", "results", "xai", "fault_isolation_report.json")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

SENSORS = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
           "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]
N_STEPS  = 50   # Riemann integration steps
N_EXPLAIN = 50  # max TP windows to explain
WINDOW   = 60
STRIDE   = 5


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


def anomaly_score_tensor(model, x_t, median_t, mad_t, weights_t):
    """Differentiable anomaly score: max weighted standardised end-error."""
    recon = model(x_t)
    err   = (recon - x_t) ** 2          # (B, T, S)
    end_err = err[:, -1, :]              # (B, S)
    end_z   = (end_err - median_t) / mad_t
    score   = (end_z * weights_t).max(dim=1).values  # (B,)
    return score


def integrated_gradients(model, x_np, baseline_np, median, mad, weights, n_steps=N_STEPS):
    """
    x_np:        (60, 9) numpy array — the window to explain
    baseline_np: (60, 9) numpy array — reference (mean normal window)
    Returns:     (60, 9) attribution array
    """
    median_t  = torch.tensor(median,  dtype=torch.float32)
    mad_t     = torch.tensor(mad,     dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)

    x        = torch.tensor(x_np,        dtype=torch.float32)
    baseline = torch.tensor(baseline_np, dtype=torch.float32)
    delta    = x - baseline

    grads = []
    for step in range(n_steps + 1):
        alpha    = step / n_steps
        interp   = (baseline + alpha * delta).unsqueeze(0).requires_grad_(True)
        score    = anomaly_score_tensor(model, interp, median_t, mad_t, weights_t)
        score.backward()
        grads.append(interp.grad.squeeze(0).detach().numpy().copy())
        model.zero_grad()

    avg_grad = np.mean(grads, axis=0)            # (60, 9)
    ig       = avg_grad * delta.numpy()          # (60, 9)
    return ig


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Integrated Gradients — Scientific XAI")
    print(f"N_STEPS={N_STEPS}, N_EXPLAIN={N_EXPLAIN}")

    batch_data = np.load(os.path.join(DATA,"batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA,"batch_faults.npy"), allow_pickle=True).item()
    op         = np.load(OP)
    mn, sc_arr = op["scaler_min"], op["scaler_scale"]
    median, mad = op["sensor_median"], op["sensor_mad"]
    weights     = op["sensor_weights"]
    threshold   = float(op["threshold"])

    X_train = np.load(os.path.join(DATA,"X_train.npy")).astype(np.float32)
    xte, yte, bte, tte = make_split(batch_data, faults, range(81,101), mn, sc_arr)

    model = GRUAutoencoderV2(9)
    ckpt  = torch.load(SAVED, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Baseline: mean normal training window (meaningful reference for IG)
    baseline = X_train.mean(axis=0)   # (60, 9)
    print(f"Baseline: mean of {len(X_train)} normal training windows")

    # Identify true-positive windows using deterministic score
    median_t  = torch.tensor(median,  dtype=torch.float32)
    mad_t     = torch.tensor(mad,     dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    all_scores = []
    with torch.no_grad():
        for s in range(0, len(xte), 256):
            z = torch.tensor(xte[s:s+256], dtype=torch.float32)
            all_scores.append(anomaly_score_tensor(model, z, median_t, mad_t, weights_t).numpy())
    scores = np.concatenate(all_scores)
    pred   = (scores > threshold).astype(int)
    tp_ix  = np.flatnonzero((pred==1) & (yte==1))[:N_EXPLAIN]
    print(f"Explaining {len(tp_ix)} true-positive windows...")

    # Compute IG for each TP window
    ig_all = np.zeros((len(tp_ix), WINDOW, 9))
    for k, idx in enumerate(tp_ix):
        ig_all[k] = integrated_gradients(
            model, xte[idx], baseline, median, mad, weights, N_STEPS)
        if (k+1) % 10 == 0:
            print(f"  {k+1}/{len(tp_ix)} done")

    # Per-sensor attribution (sum |IG| over time)
    sensor_ig = np.abs(ig_all).sum(axis=1)   # (N_explain, 9)
    mean_sensor_ig = sensor_ig.mean(axis=0)  # (9,)
    rank_idx = np.argsort(mean_sensor_ig)[::-1]

    print("\nIntegrated Gradients — Sensor Attribution Ranking:")
    for r, j in enumerate(rank_idx):
        print(f"  {r+1}. {SENSORS[j]:<35s}  mean|IG|={mean_sensor_ig[j]:.6f}")

    # Per-timestep attribution (sum |IG| over sensors)
    temporal_ig = np.abs(ig_all).sum(axis=2)  # (N_explain, 60)
    mean_temporal = temporal_ig.mean(axis=0)  # (60,)

    # Save CSVs
    sensor_rows = []
    for k, idx in enumerate(tp_ix):
        r = np.argsort(sensor_ig[k])[::-1]
        row = dict(window=int(idx), batch=int(bte[idx]),
                   time=float(tte[idx]), score=float(scores[idx]))
        for rank, j in enumerate(r):
            row[f"rank{rank+1}_sensor"] = SENSORS[j]
            row[f"rank{rank+1}_ig"] = float(sensor_ig[k, j])
        sensor_rows.append(row)
    with open(os.path.join(OUT,"ig_sensor_attributions.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sensor_rows[0]))
        w.writeheader(); w.writerows(sensor_rows)

    # Temporal rows
    time_rows = []
    for k, idx in enumerate(tp_ix):
        row = dict(window=int(idx), batch=int(bte[idx]))
        for t in range(WINDOW):
            row[f"t{t}"] = float(temporal_ig[k, t])
        time_rows.append(row)
    with open(os.path.join(OUT,"ig_temporal_attributions.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(time_rows[0]))
        w.writeheader(); w.writerows(time_rows)

    # JSON summary
    report = {
        "method": "Integrated Gradients (Sundararajan et al. 2017)",
        "n_steps": N_STEPS, "n_windows_explained": int(len(tp_ix)),
        "baseline": "mean normal training window",
        "sensor_ranking": [
            {"rank": r+1, "sensor": SENSORS[j], "mean_abs_ig": float(mean_sensor_ig[j])}
            for r, j in enumerate(rank_idx)
        ],
        "peak_attribution_timestep": int(np.argmax(mean_temporal)),
        "interpretation": (
            "IG attributes the anomaly score directly to each (timestep, sensor) "
            "input value by integrating gradients through the GRU network — not "
            "through a hand-crafted score formula. Positive IG = input value "
            "INCREASES the anomaly score relative to the normal baseline."
        )
    }
    with open(os.path.join(OUT,"ig_summary_report.json"),"w") as f:
        json.dump(report, f, indent=2)

    # Plot 1: Sensor attribution bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_s = [SENSORS[j] for j in rank_idx]
    sorted_v = mean_sensor_ig[rank_idx]
    colors   = ["#c0392b" if i==0 else "#2980b9" for i in range(9)]
    ax.barh(sorted_s[::-1], sorted_v[::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Mean |Integrated Gradient| (attribution to anomaly score)", fontsize=11)
    ax.set_title("Sensor Fault Attribution — Integrated Gradients\n"
                 "(through GRU network, not score formula)", fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"ig_sensor_barplot.png"), dpi=180)
    plt.close()

    # Plot 2: Temporal attribution heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    hm = ax.imshow(temporal_ig.T, aspect="auto", cmap="hot",
                   extent=[0, len(tp_ix)-1, WINDOW, 0])
    plt.colorbar(hm, ax=ax, label="|IG| (sum over sensors)")
    ax.set_xlabel("Explained window index"); ax.set_ylabel("Timestep within window")
    ax.set_title("Temporal Attribution Heatmap — When in the Window is the Fault?\n"
                 "(brighter = higher attribution to anomaly score)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"ig_temporal_heatmap.png"), dpi=180)
    plt.close()

    # Plot 3: IG vs SHAP comparison (if SHAP report exists)
    if os.path.exists(SHAP_JSON):
        with open(SHAP_JSON) as f:
            shap_data = json.load(f)
        shap_rank = {r["sensor"]: r["rank"] for r in shap_data["sensor_ranking"]}
        ig_rank   = {SENSORS[j]: r+1 for r, j in enumerate(rank_idx)}
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (method, ranking) in zip(axes, [("SHAP (KernelExplainer)", shap_rank),
                                                 ("Integrated Gradients", ig_rank)]):
            sensors_sorted = sorted(ranking, key=lambda s: ranking[s])
            ax.barh(range(len(sensors_sorted)), [10-ranking[s] for s in sensors_sorted],
                    color=["#c0392b" if i==0 else "#2980b9" for i in range(len(sensors_sorted))])
            ax.set_yticks(range(len(sensors_sorted)))
            ax.set_yticklabels(sensors_sorted, fontsize=9)
            ax.set_xlabel("Reverse rank (higher=more important)")
            ax.set_title(method, fontsize=11)
            ax.spines[["top","right"]].set_visible(False)
        plt.suptitle("Sensor Attribution: SHAP vs Integrated Gradients", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT,"ig_vs_shap_comparison.png"), dpi=180)
        plt.close()

    # Plot 4: Per-sensor temporal IG (mean attribution over time per sensor)
    mean_sensor_temporal = np.abs(ig_all).mean(axis=0)  # (60, 9)
    fig, ax = plt.subplots(figsize=(12, 5))
    for j in rank_idx[:5]:  # top-5 sensors only
        ax.plot(range(WINDOW), mean_sensor_temporal[:, j],
                label=SENSORS[j], linewidth=1.5)
    ax.set_xlabel("Timestep within 60-step window")
    ax.set_ylabel("Mean |IG| attribution")
    ax.set_title("Temporal Attribution Profile per Sensor (top 5)\n"
                 "Shows WHEN within the window each sensor drives the anomaly score")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"ig_temporal_per_sensor.png"), dpi=180)
    plt.close()

    print(f"\nSaved to {OUT}")
    print("  ig_sensor_barplot.png")
    print("  ig_temporal_heatmap.png")
    print("  ig_vs_shap_comparison.png")
    print("  ig_temporal_per_sensor.png")


if __name__ == "__main__":
    main()
