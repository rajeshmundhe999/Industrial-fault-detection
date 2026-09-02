# PFDI — Predictive Fault Detection and Isolation

This project detects anomalies in IndPenSim batch-process sensor data using a
GRU autoencoder.  The current validated path uses a 60-sample causal window,
training-only normalization, and normal-only validation threshold calibration.

## Quick start

Use Python 3.12 or later, then install dependencies:

```powershell
py -3.12 -m pip install -r requirements.txt
```

The included processed data and trained checkpoint are sufficient to reproduce
the results:

```powershell
py -3.12 run_project.py evaluate
py -3.12 run_project.py baselines
py -3.12 model\monitor.py --batch-id 91
py -3.12 dashboard\generate_report.py
```

## Commands

| Command | Purpose |
| --- | --- |
| `run_project.py prepare` | Rebuild processed windows from the raw CSV. |
| `run_project.py train` | Train the baseline GRU-AE using normal training batches. |
| `run_project.py evaluate` | Run the leakage-safe GRU evaluation and create deployment artifacts. |
| `run_project.py baselines` | Run PCA baselines under the identical split and calibration protocol. |
| `model/monitor.py --input batch.npy` | Score a new raw `(timesteps, 9)` sensor array. |
| `dashboard/generate_report.py` | Build `dashboard/report.html`, a portable results dashboard. |

## Validated operating point

`robust_max_val_max` is the conservative operating policy.  It takes the
maximum per-sensor, training-robust-standardized reconstruction error at the
end of each window and alarms above the largest normal validation score.

Its held-out results are ROC–AUC 0.9682, average precision 0.7299, precision
0.8366, recall 0.4776, F1 0.6081, and event recall 10/10.  See
`model/results/leakage_safe/README.md` for the protocol, results, and crucial
limitations (especially late alerts in three fault batches).

## Data contract

`model/monitor.py` expects raw measurements in exactly this column order:

1. Agitator RPM
2. Aeration rate
3. Sugar feed rate
4. Acid flow rate
5. Base flow rate
6. Heating/cooling water flow
7. Dissolved oxygen
8. pH
9. Temperature

Never refit scaling or change the score threshold using deployment/test data.
