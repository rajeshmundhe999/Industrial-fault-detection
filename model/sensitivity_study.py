"""
Hyperparameter Sensitivity Study — Window Size & Stride
========================================================
Evaluates how robust the GRU-AE v2 detector is to the choice of
sliding window size and stride. Uses the EXISTING trained model
(no retraining) — only re-windows the data and re-evaluates.

This addresses the reviewer concern: "you chose WINDOW=60, STRIDE=5
arbitrarily. Does your result hold under other settings?"

Outputs
-------
model/results/sensitivity/
  window_stride_sensitivity.csv   — full metrics for each (window, stride) combo
  sensitivity_heatmap_auc.png     — AUC heatmap over window × stride
  sensitivity_heatmap_f1.png      — F1 heatmap
  sensitivity_lineplot.png        — AUC vs window size for each stride
"""
import csv, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
SAVED = os.path.join(BASE, "model", "saved", "gru_ae_v2_best.pt")
OUT   = os.path.join(BASE, "model", "results", "sensitivity")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

WINDOWS = [20, 30, 45, 60, 90, 120]
STRIDES = [3, 5, 10]
SENSORS = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
           "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]

def scale(x, mn, sc): return x * sc + mn

def make_windows(batch_data, faults, batches, mn, sc, window, stride):
    xs, y_end, batch_ids = [], [], []
    for b in batches:
        x = scale(batch_data[str(b)], mn, sc)
        f = faults[str(b)]["fault"]
        for start in range(0, len(x) - window + 1, stride):
            end = start + window
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end-1] > 0))
            batch_ids.append(b)
    return np.asarray(xs), np.asarray(y_end), np.asarray(batch_ids)

def recon_end_errors(model, x, batch_size=256):
    chunks = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            z = torch.tensor(x[s:s+batch_size], dtype=torch.float32)
            r = model(z)
            chunks.append(((r - z)**2)[:, -1, :].numpy())
    return np.concatenate(chunks)

def run_eval(model, batch_data, faults, mn, sc, window, stride):
    xva, yva, bva = make_windows(batch_data, faults, range(71,81), mn, sc, window, stride)
    xte, yte, bte = make_windows(batch_data, faults, range(81,101), mn, sc, window, stride)

    # Clip val windows to model's trained window size if shorter
    # (pad with zeros if shorter, truncate if longer)
    def adapt_window(x, target=60):
        N, T, S = x.shape
        if T == target: return x
        if T < target:
            pad = np.zeros((N, target-T, S), dtype=np.float32)
            return np.concatenate([pad, x], axis=1)  # left-pad with zeros
        return x[:, -target:, :]  # take last `target` steps

    xva_a = adapt_window(xva, 60)
    xte_a = adapt_window(xte, 60)

    eva = recon_end_errors(model, xva_a)
    ete = recon_end_errors(model, xte_a)

    median = np.median(eva, axis=0)
    mad    = np.median(np.abs(eva - median), axis=0) * 1.4826 + 1e-12
    # Equal-weight max score (no CV weights — to isolate window effect)
    zva = (eva - median) / mad
    zte = (ete - median) / mad
    sv  = zva.max(axis=1)
    st  = zte.max(axis=1)

    # val_max threshold
    threshold = float(np.max(sv))
    pred = (st > threshold).astype(int)

    if len(np.unique(yte)) < 2:
        return None

    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    auc = roc_auc_score(yte, st)
    ap  = average_precision_score(yte, st)
    fpr = float(np.sum((pred==1)&(yte==0)) / max(1, np.sum(yte==0)))

    # Event recall
    ev_detected = 0
    for b in range(91, 101):
        bix = np.flatnonzero(bte == b)
        if pred[bix].any(): ev_detected += 1

    return dict(window=window, stride=stride, threshold=threshold,
                roc_auc=auc, average_precision=ap, f1=f,
                precision=p, recall=r, fpr=fpr,
                event_recall=ev_detected/10,
                val_windows=len(xva), test_windows=len(xte),
                test_fault_frac=float(yte.mean()))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Hyperparameter Sensitivity Study")
    print(f"Windows: {WINDOWS}")
    print(f"Strides: {STRIDES}")

    batch_data = np.load(os.path.join(DATA,"batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA,"batch_faults.npy"), allow_pickle=True).item()
    mn  = np.load(os.path.join(DATA,"scaler_min.npy"))
    sc  = np.load(os.path.join(DATA,"scaler_scale.npy"))

    model = GRUAutoencoderV2(9)
    ckpt  = torch.load(SAVED, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    results = []
    for w in WINDOWS:
        for s in STRIDES:
            if s >= w: continue
            print(f"  WINDOW={w:3d}, STRIDE={s:2d} ... ", end="", flush=True)
            r = run_eval(model, batch_data, faults, mn, sc, w, s)
            if r:
                results.append(r)
                print(f"AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  "
                      f"Recall={r['recall']:.4f}  EvRec={r['event_recall']:.1f}")
            else:
                print("skipped (insufficient labels)")

    # Save CSV
    with open(os.path.join(OUT,"window_stride_sensitivity.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader(); w.writerows(results)

    # AUC heatmap
    auc_grid = np.full((len(WINDOWS), len(STRIDES)), np.nan)
    f1_grid  = np.full((len(WINDOWS), len(STRIDES)), np.nan)
    for r in results:
        wi = WINDOWS.index(r["window"])
        si = STRIDES.index(r["stride"])
        auc_grid[wi, si] = r["roc_auc"]
        f1_grid[wi, si]  = r["f1"]

    for metric, grid, fname in [("ROC-AUC", auc_grid, "sensitivity_heatmap_auc.png"),
                                  ("F1",     f1_grid,  "sensitivity_heatmap_f1.png")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                       vmin=np.nanmin(grid)-0.02, vmax=np.nanmax(grid)+0.02)
        ax.set_xticks(range(len(STRIDES))); ax.set_xticklabels([f"stride={s}" for s in STRIDES])
        ax.set_yticks(range(len(WINDOWS))); ax.set_yticklabels([f"W={w}" for w in WINDOWS])
        for i in range(len(WINDOWS)):
            for j in range(len(STRIDES)):
                if not np.isnan(grid[i,j]):
                    ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label=metric)
        ax.set_title(f"{metric} — Sensitivity to Window Size & Stride\n"
                     f"(existing GRU-AE v2 model, no retraining)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, fname), dpi=180)
        plt.close()

    # Line plot: AUC vs window size per stride
    fig, ax = plt.subplots(figsize=(9, 5))
    for si, s in enumerate(STRIDES):
        ws = [r["window"] for r in results if r["stride"]==s]
        aucs = [r["roc_auc"] for r in results if r["stride"]==s]
        ax.plot(ws, aucs, "o-", label=f"stride={s}", linewidth=1.8)
    ax.axvline(60, color="gray", ls="--", lw=1, label="default W=60")
    ax.set_xlabel("Window size (timesteps)"); ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC vs Window Size — Hyperparameter Sensitivity\n"
                 "GRU-AE v2, existing checkpoint (no retraining)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"sensitivity_lineplot.png"), dpi=180)
    plt.close()

    # Print summary
    print("\n--- Sensitivity Summary ---")
    print(f"{'Window':>8}  {'Stride':>6}  {'AUC':>6}  {'F1':>6}  {'Recall':>6}  {'EvRec':>5}")
    for r in sorted(results, key=lambda x: (x['window'], x['stride'])):
        print(f"{r['window']:>8}  {r['stride']:>6}  {r['roc_auc']:.4f}  "
              f"{r['f1']:.4f}  {r['recall']:.4f}  {r['event_recall']:.1f}")

    aucs = [r["roc_auc"] for r in results]
    print(f"\nAUC range: {min(aucs):.4f} – {max(aucs):.4f}  "
          f"(std={np.std(aucs):.4f}, showing {'robust' if np.std(aucs)<0.02 else 'sensitive'} behaviour)")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
