"""Normal-only PCA reconstruction baselines using the same split and protocol.

No test labels are used to fit PCA, select components, or calibrate thresholds.
"""
import csv
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "model", "results", "leakage_safe")
sys.path.insert(0, os.path.join(BASE, "model"))
from leakage_safe_event_evaluation import (make_split, scale, event_metrics, WINDOW, TRAIN_BATCHES, VAL_BATCHES, TEST_BATCHES, Q)


def residual_score(model, x, endpoint_only):
    flat = x.reshape(len(x), -1)
    rec = model.inverse_transform(model.transform(flat)).reshape(x.shape)
    err = (x - rec) ** 2
    return err[:, -1, :].mean(1) if endpoint_only else err.mean((1, 2))


def metrics(y, score, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(roc_auc=roc_auc_score(y, score), average_precision=average_precision_score(y, score), precision=p, recall=r, f1=f,
                fpr=float(np.mean(pred[y == 0])))


def main():
    os.makedirs(OUT, exist_ok=True)
    bd = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
    faults = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
    mn, sc = np.load(os.path.join(DATA, "scaler_min.npy")), np.load(os.path.join(DATA, "scaler_scale.npy"))
    xtr, _, _, _ = make_split(bd, faults, TRAIN_BATCHES, mn, sc)
    xva, yva, bva, _ = make_split(bd, faults, VAL_BATCHES, mn, sc)
    xte, yte, bte, tte = make_split(bd, faults, TEST_BATCHES, mn, sc)
    assert not yva.any()
    rows = []
    # Component count is fitted from training variance alone (99% retained).
    for name, fit_x, endpoint in (("pca_window", xtr, False), ("pca_end_state", xtr[:, -1:, :], True)):
        pca = PCA(n_components=0.99, svd_solver="full", random_state=42)
        pca.fit(fit_x.reshape(len(fit_x), -1))
        sv = residual_score(pca, xva if not endpoint else xva[:, -1:, :], endpoint)
        st = residual_score(pca, xte if not endpoint else xte[:, -1:, :], endpoint)
        for policy, threshold in (("val_q99_5", float(np.percentile(sv, Q))), ("val_max", float(np.max(sv)))):
            pred = (st > threshold).astype(int)
            ev, er, nfa = event_metrics(pred, bte, tte, faults)
            rows.append(dict(method=name + "_" + policy, components=pca.n_components_, threshold=threshold,
                             validation_alarm_rate=float(np.mean(sv > threshold)), **metrics(yte, st, pred),
                             event_recall=er, normal_batch_false_alarm_rate=nfa))
    path = os.path.join(OUT, "pca_baseline_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print("%s: AUC=%.4f AP=%.4f F1=%.4f event=%.2f" % (r["method"], r["roc_auc"], r["average_precision"], r["f1"], r["event_recall"]))
    print("Saved:", path)


if __name__ == "__main__":
    main()
