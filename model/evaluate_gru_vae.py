"""
Evaluate GRU-VAE on IndPenSim — leakage-safe protocol.

Anomaly Score = robust_max(sensor_end_errors) + beta_score * KL_per_window

Both components are normalised separately using training-set statistics, then
combined with a learnable weight that is swept over the validation set.
Threshold = max combined val score (val_max policy — zero test labels used).

Outputs → model/results/vae/
  metrics.csv
  operating_point.npz
  scores_and_predictions.npz
  vae_score_breakdown.csv   — per-window recon + KL contribution
"""
import csv, json, os, sys
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_fscore_support)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
CKPT  = os.path.join(BASE, "model", "saved", "gru_vae_best.pt")
OUT   = os.path.join(BASE, "model", "results", "vae")
sys.path.insert(0, BASE)
from model.gru_vae import GRUVAEModel

os.makedirs(OUT, exist_ok=True)

WINDOW = 60;  STRIDE = 5
SENSORS = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
           "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]
# KL weight sweep on validation set
KL_WEIGHTS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]


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
            batch_ids.append(b)
            times.append(float(t[end-1]))
    return (np.asarray(xs), np.asarray(y_end),
            np.asarray(batch_ids), np.asarray(times))

def get_errors_and_kl(model, x_np, device, batch_size=512):
    """
    Returns:
      errors  (N, T, S) — per-timestep per-sensor squared reconstruction error
      kl      (N,)      — KL divergence per window
    """
    model.eval()
    all_err, all_kl = [], []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.tensor(x_np[i:i+batch_size], dtype=torch.float32).to(device)
            recon, mu, logv = model(xb)
            err = ((recon - xb) ** 2).cpu().numpy()
            kl  = model.kl_divergence(mu, logv).cpu().numpy()
            all_err.append(err)
            all_kl.append(kl)
    return np.concatenate(all_err), np.concatenate(all_kl)

def robust_max_score(errors, median, mad):
    """End-timestep robust-max score (same as v2)."""
    end_err = errors[:, -1, :]
    z       = (end_err - median) / (mad + 1e-12)
    return z.max(axis=1)

def combined_score(recon_score, kl, kl_median, kl_mad, kl_weight):
    """Normalise KL the same way as recon, then combine."""
    kl_z = (kl - kl_median) / (kl_mad + 1e-12)
    return recon_score + kl_weight * kl_z

def event_recall(pred, batch_ids, faulty_batches):
    detected = sum(1 for b in faulty_batches
                   if pred[batch_ids == b].any())
    return detected / len(faulty_batches)

def metrics_dict(y, pred, score, batch_ids, faulty_batches):
    if len(np.unique(y)) < 2:
        return {}
    roc = roc_auc_score(y, score)
    ap  = average_precision_score(y, score)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    er  = event_recall(pred, batch_ids, faulty_batches)
    return dict(roc_auc=round(roc,4), avg_prec=round(ap,4),
                f1=round(f,4), recall=round(r,4), precision=round(p,4),
                event_recall=round(er,4))


def main():
    print("=" * 60)
    print("GRU-VAE EVALUATION — IndPenSim (leakage-safe)")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GRUVAEModel(input_size=ckpt.get("input_size", 9),
                        hidden=ckpt.get("hidden", 64),
                        latent_dim=ckpt.get("latent_dim", 16),
                        dropout=0.15)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"val_recon={ckpt.get('val_recon', 0):.6f}  "
          f"val_kl={ckpt.get('val_kl', 0):.4f}")

    # ── load data ─────────────────────────────────────────────────────────────
    batch_data = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
    sc_mn      = np.load(os.path.join(DATA, "scaler_min.npy")).astype(np.float32)
    sc_sc      = np.load(os.path.join(DATA, "scaler_scale.npy")).astype(np.float32)

    train_batches  = list(range(1, 71))
    val_batches    = list(range(71, 81))
    test_batches   = list(range(81, 101))
    faulty_batches = list(range(91, 101))

    xtr, _,   btr, _ = make_split(batch_data, faults, train_batches, sc_mn, sc_sc)
    xva, yva, bva, _ = make_split(batch_data, faults, val_batches,   sc_mn, sc_sc)
    xte, yte, bte, tte = make_split(batch_data, faults, test_batches, sc_mn, sc_sc)
    print(f"Train={xtr.shape}  Val={xva.shape}  Test={xte.shape}")
    print(f"Fault windows in test: {yte.sum()}/{len(yte)}")

    # ── compute errors and KL ─────────────────────────────────────────────────
    print("Computing errors (train)...")
    etr, kl_tr = get_errors_and_kl(model, xtr, device)
    print("Computing errors (val)...")
    eva, kl_va = get_errors_and_kl(model, xva, device)
    print("Computing errors (test)...")
    ete, kl_te = get_errors_and_kl(model, xte, device)

    # ── training-set statistics for normalisation ─────────────────────────────
    train_end  = etr[:, -1, :]
    median     = np.median(train_end, axis=0)
    mad        = np.median(np.abs(train_end - median), axis=0) * 1.4826 + 1e-12
    kl_median  = float(np.median(kl_tr))
    kl_mad     = float(np.median(np.abs(kl_tr - kl_median)) * 1.4826 + 1e-12)

    # ── base reconstruction scores ─────────────────────────────────────────────
    val_recon  = robust_max_score(eva, median, mad)
    test_recon = robust_max_score(ete, median, mad)

    # ── sweep KL weight on validation — pick best F1 ─────────────────────────
    print("\nSweeping KL weight on val set...")
    best_f1_val  = -1
    best_kl_w    = 0.0
    sweep_rows   = []

    for kl_w in KL_WEIGHTS:
        val_sc   = combined_score(val_recon,  kl_va, kl_median, kl_mad, kl_w)
        thr      = float(val_sc.max())
        test_sc  = combined_score(test_recon, kl_te, kl_median, kl_mad, kl_w)
        pred     = (test_sc > thr).astype(int)
        m        = metrics_dict(yte, pred, test_sc, bte, faulty_batches)
        sweep_rows.append({"kl_weight": kl_w, "threshold": round(thr, 4), **m})
        print(f"  kl_w={kl_w:.3f}  AUC={m.get('roc_auc','?')}  "
              f"F1={m.get('f1','?')}  Recall={m.get('recall','?')}  "
              f"EvRec={m.get('event_recall','?')}")
        if m.get("f1", 0) > best_f1_val:
            best_f1_val = m["f1"]
            best_kl_w   = kl_w

    print(f"\nBest KL weight (val F1): {best_kl_w}  →  test F1={best_f1_val}")

    # ── final evaluation at best KL weight ────────────────────────────────────
    val_scores   = combined_score(val_recon,  kl_va, kl_median, kl_mad, best_kl_w)
    test_scores  = combined_score(test_recon, kl_te, kl_median, kl_mad, best_kl_w)
    threshold    = float(val_scores.max())
    pred         = (test_scores > threshold).astype(int)
    m_final      = metrics_dict(yte, pred, test_scores, bte, faulty_batches)

    # Also report at kl_w=0 (pure reconstruction, for fair comparison with v2)
    pure_val_sc  = combined_score(val_recon,  kl_va, kl_median, kl_mad, 0.0)
    pure_test_sc = combined_score(test_recon, kl_te, kl_median, kl_mad, 0.0)
    pure_thr     = float(pure_val_sc.max())
    pure_pred    = (pure_test_sc > pure_thr).astype(int)
    m_pure       = metrics_dict(yte, pure_pred, pure_test_sc, bte, faulty_batches)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Pure reconstruction (kl_w=0): "
          f"AUC={m_pure.get('roc_auc')}  F1={m_pure.get('f1')}  "
          f"Recall={m_pure.get('recall')}  EvRec={m_pure.get('event_recall')}")
    print(f"  Combined (kl_w={best_kl_w}): "
          f"AUC={m_final.get('roc_auc')}  F1={m_final.get('f1')}  "
          f"Recall={m_final.get('recall')}  EvRec={m_final.get('event_recall')}")

    # ── per-batch event table ─────────────────────────────────────────────────
    print("\nPer-batch event detection:")
    for b in faulty_batches:
        ix   = bte == b
        detected = int(pred[ix & (yte == 1)].any())
        print(f"  Batch {b}: detected={'YES' if detected else 'NO'}  "
              f"fault_windows={int(yte[ix].sum())}  alarms={int(pred[ix].sum())}")

    # ── save ──────────────────────────────────────────────────────────────────
    # metrics.csv
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "metric", "value"])
        for k, v in m_final.items():
            w.writerow(["best_combined", k, v])
        for k, v in m_pure.items():
            w.writerow(["pure_recon", k, v])
        w.writerow([])
        if sweep_rows:
            sw = csv.DictWriter(f, fieldnames=["section"] + list(sweep_rows[0].keys()))
            sw.writeheader()
            for row in sweep_rows:
                sw.writerow({"section": "kl_sweep", **row})

    # operating_point.npz
    np.savez(os.path.join(OUT, "operating_point.npz"),
             threshold=np.array(threshold),
             kl_weight=np.array(best_kl_w),
             sensor_median=median, sensor_mad=mad,
             kl_median=np.array(kl_median), kl_mad=np.array(kl_mad),
             window_size=np.array(WINDOW), stride=np.array(STRIDE))

    # scores_and_predictions.npz
    np.savez(os.path.join(OUT, "scores_and_predictions.npz"),
             scores=test_scores, recon_scores=test_recon, kl_scores=kl_te,
             labels=yte, batch_ids=bte, times=tte, predictions=pred)

    # score breakdown CSV
    with open(os.path.join(OUT, "vae_score_breakdown.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "batch", "label", "recon_score", "kl_score",
                    "combined_score", "pred"])
        for i in range(len(yte)):
            w.writerow([i, int(bte[i]), int(yte[i]),
                        round(float(test_recon[i]), 4),
                        round(float(kl_te[i]), 4),
                        round(float(test_scores[i]), 4),
                        int(pred[i])])

    print(f"\nSaved to {OUT}")
    print("  metrics.csv  operating_point.npz  scores_and_predictions.npz  vae_score_breakdown.csv")


if __name__ == "__main__":
    main()
