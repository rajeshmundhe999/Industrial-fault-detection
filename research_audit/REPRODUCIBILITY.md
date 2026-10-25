# Reproducibility Guide — PFDI Project
**Purpose**: Step-by-step instructions to reproduce every reported result from scratch.

---

## Environment Setup

```powershell
# Python version
py -3.12 --version  # Expected: Python 3.12.x

# Core packages (already installed)
py -3.12 -m pip install torch numpy scikit-learn matplotlib shap captum

# Dashboard (install separately)
py -3.12 -m pip install streamlit plotly pandas

# Optional (KMeans used as fallback if not available)
py -3.12 -m pip install hdbscan umap-learn
```

**Working directory**: `c:\000 VIT\ETHANOL\pfdi_project`  
**Shell**: PowerShell — always prepend `$env:PYTHONIOENCODING="utf-8";`  
**GPU**: Not required. All results reproducible on CPU (training: ~2–5 hrs per model).

---

## Step 1: Data Preprocessing

The raw IndPenSim data is already preprocessed. If re-running from scratch:

```powershell
$env:PYTHONIOENCODING="utf-8"; py -3.12 data/preprocess.py
```

Expected output files in `data/processed/`:
- `batch_data.npy` — shape (100, 500, 9)
- `batch_faults.npy` — shape (100, 500)
- `X_train.npy` — shape (15216, 60, 9)
- `X_val.npy` — shape (2122, 60, 9)
- `X_test.npy` — shape (4349, 60, 9)
- `y_test.npy`
- `scaler_min.npy`, `scaler_scale.npy`

**Data split**: Train = batches 1–70, Val = 71–80, Test = 81–100  
**Scaler**: MinMaxScaler fitted on training data ONLY

---

## Step 2: Train Primary Model (GRU-AE v2)

> [!NOTE]
> The trained checkpoint is already saved at `model/saved/gru_ae_v2_best.pt` (val_loss=8.9e-5, epoch 261).
> Skip this step unless re-training from scratch.

```powershell
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/train_gru_v2.py
```

Expected: Best val_loss ≈ 0.000089, ~261 epochs (may vary with seed)  
Saves: `model/saved/gru_ae_v2_best.pt`

---

## Step 3: Evaluate Primary Model

```powershell
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/evaluate_gru_v2.py
```

Saves: `model/results/leakage_safe_v2/scores_and_predictions.npz`

**Verified output** (re-computed from saved NPZ):
- `end_robust_mean_val_max_score`: AUC=0.9591, F1=0.6393, Recall=0.5821, Precision=0.7091, EventRecall=1.000
- `end_robust_max_val_q99_5_score`: AUC=0.9584, F1=0.7138, Recall=0.7910, Precision=0.6503, EventRecall=1.000

---

## Step 4: Baseline Models

```powershell
# LSTM-AE
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/train_lstm_ae.py

# TranAD
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/train_tranad_original.py

# GRU-VAE
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/train_gru_vae.py
```

---

## Step 5: Supporting Experiments

```powershell
# Sensitivity study (18 configs, ~30 min)
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/run_sensitivity.py

# Multi-scale ensemble
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/run_multiscale.py

# SHAP explanations
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/run_shap.py

# Integrated Gradients
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/run_ig.py

# Early detection (trend alarm)
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/run_early_detection.py

# MC-Dropout uncertainty
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/run_uncertainty.py

# Latent clustering
$env:PYTHONIOENCODING="utf-8"; py -3.12 model/run_fault_isolation.py
```

---

## Step 6: TEP Benchmark

```powershell
# Preprocess TEP (raw data already in tep/raw/)
$env:PYTHONIOENCODING="utf-8"; py -3.12 tep/preprocess_tep.py

# Train TEP model (original, 341 windows)
$env:PYTHONIOENCODING="utf-8"; py -3.12 tep/train_tep.py

# Evaluate TEP original
$env:PYTHONIOENCODING="utf-8"; py -3.12 tep/evaluate_tep.py

# Train TEP augmented (2046 windows, ~4 hrs CPU)
$env:PYTHONIOENCODING="utf-8"; py -3.12 tep/train_tep_aug.py

# Evaluate TEP augmented
$env:PYTHONIOENCODING="utf-8"; py -3.12 tep/evaluate_tep_aug.py
```

---

## Step 7: Architecture Ablation (NEW)

```powershell
$env:PYTHONIOENCODING="utf-8"; py -3.12 research_audit/ablation/train_ablation.py
```

Expected runtime: ~9 hours total (3 variants × ~3 hrs each on CPU)  
Results: `research_audit/ablation/results/ablation_results.csv`

---

## Step 8: Batch 93 Investigation

```powershell
$env:PYTHONIOENCODING="utf-8"; py -3.12 research_audit/plot_batch93.py
```

Output: `research_audit/figures/batch93_score_curve.png`

---

## Step 9: Dashboard

```powershell
py -3.12 -m streamlit run dashboard/app.py
```

Open browser at `http://localhost:8501`

---

## Known Issues

| Issue | Description | Workaround |
|---|---|---|
| `hdbscan` not installed | Latent clustering falls back to KMeans | Install with `pip install hdbscan` |
| `umap-learn` not installed | UMAP projection not available | Install or use PCA fallback |
| GPU not available | Training on CPU only | ~2–5x slower; all results reproducible |
| TranAD scoring | Non-native scoring formula used | Documented in RESULTS_AUDIT.md |
| Trend alarm batch 93 | Spurious false alarm at window 11 | Documented in PAPER_CLAIMS_AUDIT.md |

---

## File Checksums (Key Results)

Run to verify integrity of primary results:

```powershell
$env:PYTHONIOENCODING="utf-8"; py -3.12 -c "
import numpy as np
d = np.load('model/results/leakage_safe_v2/scores_and_predictions.npz')
s = d['end_robust_mean_val_max_score']
print(f'Primary score array: mean={s.mean():.4f}, max={s.max():.4f}, len={len(s)}')
# Expected: mean≈4.8, max≈338, len=4349
"
```
