"""Leakage-safe GRU-AE evaluation at the causal (window-end) decision time.

This is deliberately separate from the earlier label-search experiments.  It
uses the original, correctly MinMax-scaled data, derives every threshold from
normal validation batches only, and never uses test labels to choose a score,
threshold, sensor weight, or persistence rule.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")
SAVED = os.path.join(BASE, "model", "saved", "gru_ae_best.pt")
OUT = os.path.join(BASE, "model", "results", "leakage_safe")
PLOTS = os.path.join(BASE, "model", "plots", "leakage_safe")
sys.path.insert(0, BASE)
from model.gru_ae import GRUAutoencoder

WINDOW, STRIDE = 60, 5
TRAIN_BATCHES, VAL_BATCHES, TEST_BATCHES = range(1, 71), range(71, 81), range(81, 101)
SENSORS = ["Agitator RPM", "Aeration rate", "Sugar feed rate", "Acid flow rate", "Base flow rate", "Heating/cooling water flow", "Dissolved oxygen", "pH", "Temperature"]
Q = 99.5  # pre-specified validation-normal window false-alarm target: approximately 0.5%


def scale(x, scaler_min, scaler_scale):
    # sklearn MinMaxScaler.transform: X * scale_ + min_.  Do not invert this.
    return x * scaler_scale + scaler_min


def make_split(batch_data, faults, batches, scaler_min, scaler_scale):
    xs, y_end, batch_ids, times = [], [], [], []
    for b in batches:
        x = scale(batch_data[str(b)], scaler_min, scaler_scale)
        f, t = faults[str(b)]["fault"], faults[str(b)]["time"]
        for start in range(0, len(x) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end - 1] > 0))
            batch_ids.append(b)
            times.append(float(t[end - 1]))
    return np.asarray(xs), np.asarray(y_end), np.asarray(batch_ids), np.asarray(times)


def reconstruction_errors(model, x):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            z = torch.tensor(x[start:start + 512], dtype=torch.float32)
            chunks.append(((model(z) - z) ** 2).numpy())
    return np.concatenate(chunks)


def smooth_by_batch(scores, batch_ids, width=3):
    out = np.empty_like(scores)
    for b in np.unique(batch_ids):
        ix = np.flatnonzero(batch_ids == b)
        # Strictly trailing (not centred): an alarm cannot inspect future windows.
        for j in range(len(ix)):
            out[ix[j]] = np.mean(scores[ix[max(0, j - width + 1):j + 1]])
    return out


def persistence(pred, batch_ids, required=2, width=3):
    out = np.zeros_like(pred)
    for b in np.unique(batch_ids):
        ix = np.flatnonzero(batch_ids == b)
        for j in range(len(ix)):
            lo = max(0, j - width + 1)
            out[ix[j]] = int(np.sum(pred[ix[lo:j + 1]]) >= required)
    return out


def point_metrics(y, score, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(roc_auc=roc_auc_score(y, score), average_precision=average_precision_score(y, score), precision=p, recall=r, f1=f,
                fpr=float(np.sum((pred == 1) & (y == 0)) / max(1, np.sum(y == 0))))


def event_metrics(pred, batch_ids, times, faults):
    rows, detected, false_alarm_batches = [], 0, 0
    for b in TEST_BATCHES:
        ix = np.flatnonzero(batch_ids == b)
        f, t = faults[str(b)]["fault"], faults[str(b)]["time"]
        fault_ix = np.flatnonzero(f > 0)
        fault_start = float(t[fault_ix[0]]) if len(fault_ix) else np.nan
        alarms = times[ix][pred[ix] == 1]
        if len(fault_ix):
            post = alarms[alarms >= fault_start]
            hit = len(post) > 0
            detected += int(hit)
            delay = float(post[0] - fault_start) if hit else np.nan
            pre_alarm = bool(np.any(alarms < fault_start))
        else:
            hit, delay, pre_alarm = False, np.nan, len(alarms) > 0
            false_alarm_batches += int(pre_alarm)
        rows.append(dict(batch=b, faulty=int(len(fault_ix) > 0), fault_start=fault_start, detected=int(hit), detection_delay=delay,
                         alarm_windows=len(alarms), pre_fault_alarm=int(pre_alarm)))
    return rows, detected / 10.0, false_alarm_batches / 10.0


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    batch_data = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
    faults = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
    mn, sc = np.load(os.path.join(DATA, "scaler_min.npy")), np.load(os.path.join(DATA, "scaler_scale.npy"))
    xtr, _, _, _ = make_split(batch_data, faults, TRAIN_BATCHES, mn, sc)
    xva, yva, bva, _ = make_split(batch_data, faults, VAL_BATCHES, mn, sc)
    xte, yte, bte, tte = make_split(batch_data, faults, TEST_BATCHES, mn, sc)
    assert not yva.any(), "Validation must remain normal-only for this protocol"
    model = GRUAutoencoder(9)
    model.load_state_dict(torch.load(SAVED, map_location="cpu", weights_only=False)["model_state_dict"])
    etr, eva, ete = reconstruction_errors(model, xtr), reconstruction_errors(model, xva), reconstruction_errors(model, xte)
    # Scores are causal: only reconstruction error at the window end is used.
    train_sensor, val_sensor, test_sensor = etr[:, -1, :], eva[:, -1, :], ete[:, -1, :]
    median = np.median(train_sensor, axis=0); mad = np.median(np.abs(train_sensor - median), axis=0) * 1.4826 + 1e-12
    zva, zte = (val_sensor - median) / mad, (test_sensor - median) / mad
    candidates = {
        "end_mse": (val_sensor.mean(1), test_sensor.mean(1)),
        "robust_mean": (zva.mean(1), zte.mean(1)),
        "robust_max": (zva.max(1), zte.max(1)),
    }
    rows, selected = [], {}
    for name, (sv, st) in candidates.items():
        for smoothed in (False, True):
            score_label = name + ("_causalmean3" if smoothed else "")
            vv, tt = (smooth_by_batch(sv, bva), smooth_by_batch(st, bte)) if smoothed else (sv, st)
            # Two pre-declared normal-only operating policies.  The second is
            # intentionally conservative for batch operations: no observed
            # validation window may trigger an alarm.
            for policy, threshold in (("val_q99_5", float(np.percentile(vv, Q))),
                                      ("val_max", float(np.max(vv)))):
                label = score_label + "_" + policy
                raw_pred = (tt > threshold).astype(int)
                pred = persistence(raw_pred, bte) if smoothed else raw_pred
                m = point_metrics(yte, tt, pred)
                events, event_recall, normal_batch_fa = event_metrics(pred, bte, tte, faults)
                rows.append(dict(method=label, threshold=threshold, validation_alarm_rate=float(np.mean(vv > threshold)), **m,
                                 event_recall=event_recall, normal_batch_false_alarm_rate=normal_batch_fa))
                selected[label] = (tt, pred, events)
    write_csv(os.path.join(OUT, "metrics.csv"), rows)
    # Sensor diagnosis is explicitly test descriptive analysis, never used in score selection.
    diag = []
    for j, s in enumerate(SENSORS):
        diag.append(dict(sensor=s, train_median=float(median[j]), train_robust_scale=float(mad[j]), val_q995=float(np.percentile(val_sensor[:,j], Q)),
                         test_normal_mean=float(test_sensor[yte == 0,j].mean()), test_fault_mean=float(test_sensor[yte == 1,j].mean())))
    write_csv(os.path.join(OUT, "sensor_diagnostics_test_descriptive_only.csv"), diag)
    # Conservative operational candidate, chosen solely from the validation
    # normal-only policy, rather than by its test labels or F1.
    primary = "robust_max_val_max"
    write_csv(os.path.join(OUT, "primary_event_details.csv"), selected[primary][2])
    primary_threshold = next(r["threshold"] for r in rows if r["method"] == primary)
    # Deployment artifact: everything needed for a raw-data monitor is learned
    # from training data plus normal validation data, never test labels.
    np.savez(os.path.join(OUT, "operating_point.npz"), scaler_min=mn, scaler_scale=sc,
             sensor_median=median, sensor_robust_scale=mad, threshold=primary_threshold,
             window_size=WINDOW, stride=STRIDE)
    with open(os.path.join(OUT, "operating_point.json"), "w") as f:
        json.dump({"method": primary, "threshold": primary_threshold,
                   "threshold_policy": "maximum normal validation score",
                   "score": "maximum training-robust-standardized squared reconstruction error at window end",
                   "window_size": WINDOW, "stride": STRIDE}, f, indent=2)
    np.savez(os.path.join(OUT, "scores_and_predictions.npz"), y_end=yte, batch_ids=bte, end_times=tte,
             **{k + "_score": v[0] for k,v in selected.items()}, **{k + "_pred": v[1] for k,v in selected.items()})
    score, pred, _ = selected[primary]
    plt.figure(figsize=(12, 4)); plt.scatter(np.arange(len(score)), score, c=yte, s=7, cmap="coolwarm", label="window end label")
    plt.scatter(np.flatnonzero(pred), score[pred == 1], facecolors="none", edgecolors="black", s=24, label="alarm")
    plt.title("Primary causal score: robust mean, 3-window trailing mean, 2-of-3 persistence")
    plt.xlabel("test window index"); plt.ylabel("score"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "primary_score_overview.png"), dpi=180); plt.close()
    print("Protocol: train-only normalization; normal-only validation calibration (q%.1f and maximum)." % Q)
    print("Validation windows:", len(xva), "positive labels:", int(yva.sum()))
    for r in rows: print(r["method"], "F1=%.4f AUC=%.4f AP=%.4f event_recall=%.2f" % (r["f1"], r["roc_auc"], r["average_precision"], r["event_recall"]))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
