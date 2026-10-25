"""
Multi-Scale Ensemble Scorer
============================
Uses the EXISTING trained GRU-AE v2 model (no retraining).
Scores each test window at three window sizes: W=45, W=60, W=90.
Normalises each scale's scores with its own val statistics, then averages.

Why this helps:
  W=45 → catches fast faults (e.g. sudden step changes)
  W=60 → current optimum (our primary model window)
  W=90 → catches slow gradual faults (drift, ramp failures)
  Average → robust to timescale of fault

Threshold = max of ensemble val scores (val_max policy — no test labels).

Outputs → model/results/multiscale/
  metrics.csv
  per_scale_vs_ensemble.csv
  ensemble_score_curves.png
"""
import csv, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_fscore_support)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
CKPT  = os.path.join(BASE, "model", "saved", "gru_ae_v2_best.pt")
OUT   = os.path.join(BASE, "model", "results", "multiscale")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

os.makedirs(OUT, exist_ok=True)

WINDOWS = [45, 60, 90]
STRIDE  = 5
SENSORS = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
           "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]


def scale(x, mn, sc): return x * sc + mn

def make_windows(batch_data, faults, batches, mn, sc, window):
    xs, y_end, batch_ids, times = [], [], [], []
    for b in batches:
        x = scale(batch_data[str(b)], mn, sc)
        f = faults[str(b)]["fault"]
        t = faults[str(b)]["time"]
        for start in range(0, len(x) - window + 1, STRIDE):
            end = start + window
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end-1] > 0))
            batch_ids.append(b)
            times.append(float(t[end-1]))
    return (np.asarray(xs), np.asarray(y_end),
            np.asarray(batch_ids), np.asarray(times))

def get_errors(model, x_np, device, batch_size=512):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb    = torch.tensor(x_np[i:i+batch_size], dtype=torch.float32).to(device)
            recon = model(xb)
            chunks.append(((recon - xb) ** 2).cpu().numpy())
    return np.concatenate(chunks)   # (N, T, S)

def robust_max_score(errors, median, mad):
    end_err = errors[:, -1, :]
    z       = (end_err - median) / (mad + 1e-12)
    return z.max(axis=1)

def event_recall(pred, batch_ids, faulty_batches):
    return sum(1 for b in faulty_batches if pred[batch_ids == b].any()) / len(faulty_batches)

def eval_metrics(y, pred, score, batch_ids, faulty_batches):
    roc = roc_auc_score(y, score)
    ap  = average_precision_score(y, score)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    er  = event_recall(pred, batch_ids, faulty_batches)
    return dict(roc_auc=round(roc,4), avg_prec=round(ap,4),
                f1=round(f,4), recall=round(r,4), precision=round(p,4),
                event_recall=round(er,4))


def main():
    print("=" * 60)
    print("MULTI-SCALE ENSEMBLE SCORING")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model (input_size=9, works for any window length) ───────────────
    ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GRUAutoencoderV2(input_size=9, dropout=0.15).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded GRU-AE v2: epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', 0):.6f}")

    # ── load raw data ─────────────────────────────────────────────────────────
    batch_data = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
    sc_mn      = np.load(os.path.join(DATA, "scaler_min.npy")).astype(np.float32)
    sc_sc      = np.load(os.path.join(DATA, "scaler_scale.npy")).astype(np.float32)

    train_batches  = list(range(1, 71))
    val_batches    = list(range(71, 81))
    test_batches   = list(range(81, 101))
    faulty_batches = list(range(91, 101))

    # ── per-scale scoring ─────────────────────────────────────────────────────
    scale_val_scores  = {}
    scale_test_scores = {}
    scale_test_y      = {}
    scale_test_bids   = {}
    scale_test_times  = {}

    for W in WINDOWS:
        print(f"\n--- Window={W} ---")
        xtr, _,   _, _      = make_windows(batch_data, faults, train_batches, sc_mn, sc_sc, W)
        xva, _,   _, _      = make_windows(batch_data, faults, val_batches,   sc_mn, sc_sc, W)
        xte, yte, bte, tte  = make_windows(batch_data, faults, test_batches,  sc_mn, sc_sc, W)

        etr = get_errors(model, xtr, device)
        eva = get_errors(model, xva, device)
        ete = get_errors(model, xte, device)

        # training stats
        train_end = etr[:, -1, :]
        median    = np.median(train_end, axis=0)
        mad       = np.median(np.abs(train_end - median), axis=0) * 1.4826 + 1e-12

        vs  = robust_max_score(eva, median, mad)
        ts  = robust_max_score(ete, median, mad)

        # normalise to [0,1] range using val max (prevents one scale dominating)
        vs_max = vs.max() + 1e-12
        vs_norm = vs / vs_max
        ts_norm = ts / vs_max   # same scale factor for test

        scale_val_scores[W]  = vs_norm
        scale_test_scores[W] = ts_norm
        scale_test_y[W]      = yte
        scale_test_bids[W]   = bte
        scale_test_times[W]  = tte

        # Single-scale metrics
        thr  = float(vs_norm.max())
        pred = (ts_norm > thr).astype(int)
        m    = eval_metrics(yte, pred, ts_norm, bte, faulty_batches)
        print(f"  Single-scale W={W}: AUC={m['roc_auc']}  F1={m['f1']}  "
              f"Recall={m['recall']}  EvRec={m['event_recall']}")

    # ── ensemble: average normalised scores aligned by time ───────────────────
    # All scales score the same test batches but at different granularities.
    # We align by (batch, end_time) → use the W=60 index as reference and
    # interpolate other scales' scores to the same time axis.
    print("\n--- Building ensemble ---")

    # Reference: W=60
    ref_y    = scale_test_y[60]
    ref_bids = scale_test_bids[60]
    ref_t    = scale_test_times[60]

    ensemble_scores = np.zeros(len(ref_y))
    n_contrib       = np.zeros(len(ref_y))

    for W in WINDOWS:
        yt   = scale_test_y[W]
        bt   = scale_test_bids[W]
        tt   = scale_test_times[W]
        sc   = scale_test_scores[W]

        for b in np.unique(ref_bids):
            ref_ix  = np.flatnonzero(ref_bids == b)
            this_ix = np.flatnonzero(bt == b)
            if len(this_ix) == 0:
                continue
            # Interpolate this scale's scores to reference time axis
            ref_t_b  = ref_t[ref_ix]
            this_t_b = tt[this_ix]
            this_s_b = sc[this_ix]
            interp   = np.interp(ref_t_b, this_t_b, this_s_b)
            ensemble_scores[ref_ix] += interp
            n_contrib[ref_ix]       += 1

    ensemble_scores /= np.maximum(n_contrib, 1)

    # Threshold from ensemble val scores
    ref_val_y    = np.zeros(len(scale_val_scores[60]))
    ref_val_bids_arr = list(range(71, 81))
    _, val_y_60, val_bids_60, val_t_60 = make_windows(
        batch_data, faults, list(range(71,81)), sc_mn, sc_sc, 60)

    ensemble_val = np.zeros(len(val_y_60))
    n_v          = np.zeros(len(val_y_60))
    for W in WINDOWS:
        _, _, bva, tva = make_windows(
            batch_data, faults, list(range(71,81)), sc_mn, sc_sc, W)
        vs_norm = scale_val_scores[W]
        for b in np.unique(val_bids_60):
            ri = np.flatnonzero(val_bids_60 == b)
            vi = np.flatnonzero(bva == b)
            if len(vi) == 0:
                continue
            interp = np.interp(val_t_60[ri], tva[vi], vs_norm[vi])
            ensemble_val[ri] += interp
            n_v[ri]          += 1
    ensemble_val /= np.maximum(n_v, 1)

    ens_threshold = float(ensemble_val.max())
    ens_pred      = (ensemble_scores > ens_threshold).astype(int)
    m_ens         = eval_metrics(ref_y, ens_pred, ensemble_scores,
                                 ref_bids, faulty_batches)

    print(f"\nEnsemble (W=45+60+90): AUC={m_ens['roc_auc']}  F1={m_ens['f1']}  "
          f"Recall={m_ens['recall']}  Precision={m_ens['precision']}  "
          f"EvRec={m_ens['event_recall']}")

    # Also report reference W=60 alone (for comparison)
    thr60  = float(scale_val_scores[60].max())
    pred60 = (scale_test_scores[60] > thr60).astype(int)
    m60    = eval_metrics(ref_y, pred60, scale_test_scores[60],
                          ref_bids, faulty_batches)
    print(f"Single W=60 only:       AUC={m60['roc_auc']}  F1={m60['f1']}  "
          f"Recall={m60['recall']}  Precision={m60['precision']}  "
          f"EvRec={m60['event_recall']}")

    # ── save ──────────────────────────────────────────────────────────────────
    rows = [{"method": f"single_W{W}", **eval_metrics(
                scale_test_y[W],
                (scale_test_scores[W] > scale_val_scores[W].max()).astype(int),
                scale_test_scores[W], scale_test_bids[W], faulty_batches)}
             for W in WINDOWS]
    rows.append({"method": "ensemble_W45_60_90", **m_ens})

    with open(os.path.join(OUT, "per_scale_vs_ensemble.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + list(rows[0].keys())[1:])
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "metric", "value"])
        for k, v in m_ens.items():
            w.writerow(["ensemble", k, v])
        for k, v in m60.items():
            w.writerow(["single_W60", k, v])

    np.savez(os.path.join(OUT, "ensemble_scores.npz"),
             scores=ensemble_scores, labels=ref_y, batch_ids=ref_bids,
             times=ref_t, predictions=ens_pred, threshold=np.array(ens_threshold))

    # ── plot: score curves for faulty batches ─────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, b in zip(axes.flat, faulty_batches):
        ix = ref_bids == b
        t  = ref_t[ix]
        s  = ensemble_scores[ix]
        lb = ref_y[ix]
        ax.fill_between(t, 0, lb * s.max() * 1.1, alpha=0.15,
                        color="red", label="fault region")
        ax.plot(t, s, color="steelblue", lw=1.2, label="ensemble")
        # individual scales
        for W, col in zip(WINDOWS, ["#f4a261", "#2a9d8f", "#e76f51"]):
            ix_w = scale_test_bids[W] == b
            ax.plot(scale_test_times[W][ix_w], scale_test_scores[W][ix_w],
                    color=col, lw=0.7, alpha=0.6, label=f"W={W}")
        ax.axhline(ens_threshold, color="red", ls="--", lw=1)
        ax.set_title(f"Batch {b}", fontsize=9)
        ax.set_xlabel("time (h)", fontsize=8); ax.set_ylabel("norm score", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    plt.suptitle("Multi-Scale Ensemble vs Individual Scale Scores — Faulty Batches", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ensemble_score_curves.png"), dpi=160)
    plt.close()

    print(f"\nSaved to {OUT}")
    print("  metrics.csv  per_scale_vs_ensemble.csv  ensemble_scores.npz  ensemble_score_curves.png")


if __name__ == "__main__":
    main()
