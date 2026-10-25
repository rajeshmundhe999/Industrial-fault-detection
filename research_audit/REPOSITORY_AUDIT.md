# Repository Audit — PFDI Project
**Auditor**: Antigravity AI  
**Date**: 2026-10-19  
**Project Root**: `c:\000 VIT\ETHANOL\pfdi_project`

---

## 1. Repository Structure

```
pfdi_project/
├── data/
│   └── processed/
│       ├── batch_data.npy         # (100, 500, 9) raw per-batch sensor readings
│       ├── batch_faults.npy       # (100, 500) fault labels
│       ├── scaler_min.npy         # MinMaxScaler min (fitted on train batches 1-70)
│       └── scaler_scale.npy       # MinMaxScaler scale
├── model/
│   ├── gru_ae_v2.py               # Primary model: BiGRU+Attention+per-sensor heads
│   ├── gru_vae.py                 # Variational AE variant
│   ├── lstm_ae.py                 # LSTM-AE baseline
│   ├── tranad_original.py         # TranAD baseline
│   ├── train_gru_v2.py            # Training script (GRU-AE v2)
│   ├── train_gru_vae.py           # Training script (VAE)
│   ├── train_lstm_ae.py           # Training script (LSTM-AE)
│   ├── train_tranad_original.py   # Training script (TranAD)
│   ├── evaluate_gru_v2.py         # Evaluation with multi-policy scoring
│   ├── leakage_safe_event_evaluation.py  # Event-level evaluation helper
│   ├── run_sensitivity.py         # 18-config sensitivity study
│   ├── run_multiscale.py          # Multi-scale ensemble (W=45,60,90)
│   ├── run_shap.py                # SHAP explanations
│   ├── run_ig.py                  # Integrated Gradients
│   ├── run_early_detection.py     # Trend-based early warning
│   ├── run_uncertainty.py         # MC-Dropout
│   ├── run_fault_isolation.py     # Latent-space clustering
│   ├── saved/                     # Checkpoints
│   │   ├── gru_ae_v2_best.pt      # 49k params, BiGRU+Attn, val_loss=8.9e-5, epoch 261
│   │   ├── gru_vae_best.pt        # 70k params, val_recon=0.002086, epoch 88
│   │   ├── lstm_ae_best.pt        # 14.8k params, val_loss=0.001475, epoch ~280
│   │   ├── tranad_original_best.pt
│   │   └── paper_protocol_gru.pt  # Older checkpoint — not primary
│   └── results/
│       ├── leakage_safe_v2/       # PRIMARY RESULT SET
│       │   ├── scores_and_predictions.npz   # 70+ scoring policies stored
│       │   └── operating_point.npz          # threshold=56.37, W=60, stride=5
│       ├── leakage_safe/          # Earlier run (pre-v2) — superseded
│       ├── sensitivity/           # 18 (W, stride) configs
│       ├── multiscale/            # Multi-scale ensemble results
│       ├── early_detection/       # Trend alarm vs static alarm per-batch
│       ├── fault_isolation/       # Latent clusters, SHAP, IG
│       ├── xai/                   # SHAP plots
│       ├── integrated_gradients/  # IG results
│       ├── uncertainty/           # MC-Dropout results
│       ├── vae/
│       ├── lstm_ae/
│       └── tranad_leakage_safe/
├── tep/
│   ├── preprocess_tep.py
│   ├── train_tep.py
│   ├── train_tep_aug.py
│   ├── evaluate_tep.py
│   ├── evaluate_tep_aug.py
│   ├── raw/                      # d00.dat ... d21_te.dat (all 43 files present)
│   ├── processed/                # X_train, X_val, X_test, y_test, batch_ids, scalers
│   ├── saved/
│   │   ├── gru_ae_v2_tep_best.pt       # 341-window original
│   │   └── gru_ae_v2_tep_aug_best.pt   # 2046-window augmented
│   └── results/
│       ├── metrics.csv           # Original TEP results
│       └── aug/metrics.csv       # Augmented TEP results
├── shared/
│   ├── config.py
│   ├── interfaces.py
│   └── __init__.py
├── research_audit/               # (THIS DIRECTORY — created 2026-10-19)
└── dashboard/                    # Streamlit app (to be built)
```

---

## 2. Data Protocol (Verified)

| Split | Batches | Windows | Purpose |
|---|---|---|---|
| Train | 1–70 (normal only) | 15,216 | Model training |
| Validation | 71–80 (normal only) | 2,122 | Threshold calibration ONLY |
| Test | 81–100 (81–90 normal, 91–100 faulty) | 4,349 | Evaluation — never touched before |

- **Scaler**: MinMaxScaler fitted on train batches only → applied to val and test
- **Thresholds**: `val_max` = max of 2,122 val window scores; `val_q99.5` = 99.5th percentile of val scores
- **No test labels used** in training, validation, threshold calibration, or checkpoint selection ✅

---

## 3. Models

### GRU-AE v2 (Primary)
- **Architecture**: Bidirectional GRU encoder → Attention → Linear latent → GRU decoder → 9 per-sensor linear heads
- **Parameters**: ~49,000
- **Input**: Windows of shape (60, 9), stride=5
- **Training**: MSE loss, AdamW, cosine LR schedule, Gaussian noise augmentation, patience=40
- **Checkpoint**: `model/saved/gru_ae_v2_best.pt`, val_loss=8.9×10⁻⁵, epoch 261

### GRU-VAE (Comparison)
- **Architecture**: Bidirectional GRU encoder → μ/logvar → reparametrize → GRU decoder
- **Parameters**: ~70,000
- **Training**: ELBO loss (recon + β×KL), β-annealing from 0→0.1 over first 50 epochs

### LSTM-AE (Baseline)
- **Parameters**: ~14,800
- **Checkpoint**: `model/saved/lstm_ae_best.pt`

### TranAD (Baseline)
- **Note**: Evaluated using GRU-AE-style robust_max scoring — may understate TranAD's true performance (see RESULTS_AUDIT.md)

---

## 4. Key Files and Their Roles

| File | Purpose | Status |
|---|---|---|
| `evaluate_gru_v2.py` | Generates all 70+ scoring policies for GRU-AE v2 | ✅ Verified |
| `leakage_safe_event_evaluation.py` | Per-batch event-level evaluation | ✅ Verified |
| `run_sensitivity.py` | 18-config (W×stride) sweep | ✅ Complete |
| `run_multiscale.py` | 3-scale ensemble | ✅ Complete |
| `run_shap.py` | SHAP attributions | ✅ Complete |
| `run_ig.py` | Integrated Gradients | ✅ Complete |
| `run_early_detection.py` | Trend-based early warning | ✅ Complete |
| `run_uncertainty.py` | MC-Dropout | ✅ Complete |
| `run_fault_isolation.py` | Latent clustering | ✅ Complete |

---

## 5. Missing / Not Found

| Item | Status |
|---|---|
| Architecture ablation variants (no-attention, unidirectional, shared-head) | ❌ Not yet implemented |
| Multi-seed results (3 seeds for GRU-AE v2) | ❌ Not yet run |
| Proper early-warning evaluation at fixed FPR operating points | ❌ Not done |
| Streamlit dashboard | ❌ Not yet built |
| Paper draft | ❌ Not written |
| Literature review | 🔄 In progress (subagent) |

---

## 6. Dependencies (Verified)

- Python 3.12 (`py -3.12`)
- PyTorch (CPU only — no CUDA)
- NumPy, SciPy, scikit-learn, matplotlib
- SHAP 0.54.1
- hdbscan — NOT installed (KMeans used as fallback in fault_isolation)
- captum — for Integrated Gradients
- streamlit — to be installed for dashboard

---

## 7. Environment Notes

- All runs use `$env:PYTHONIOENCODING="utf-8"` on Windows PowerShell
- GPU unavailable — training on CPU (~2–5 hrs per model)
- Paths use Windows backslash; scripts use `os.path.join` or raw strings

---

## 8. TEP Benchmark

- Data: Braatz Lab TEP dataset (d00–d21, training + test splits)
- All 43 raw files present in `tep/raw/`
- 22 measured process variables used (columns 0–21)
- Faults 3 and 9 excluded (unmeasured/undetectable)
- Training windows: 341 (original), 2,046 (augmented with phase shifts + noise)
- Test: d00_te (normal) + d01–d21_te (faults 1–21 excluding 3,9)
- Fault injection at step 160 of each 960-step test simulation

---

*Audit Status*: Complete as of 2026-10-19. Architecture ablations pending (see EXPERIMENT_PLAN.md).
