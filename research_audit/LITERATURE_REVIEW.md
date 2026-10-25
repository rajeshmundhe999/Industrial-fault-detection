# Literature Review — PFDI Project
**Prepared by**: Antigravity AI Research Subagent  
**Date**: 2026-10-19  
**Focus**: Deep learning fault detection for industrial time-series, 2019–2026

> [!CAUTION]
> Citations marked ⚠️ should be verified against actual publication records before inclusion in a paper. DOIs should be confirmed.

---

## Section 1: GRU/LSTM Autoencoder Anomaly Detection

| # | Authors | Title | Venue | Year | Dataset | AUC/F1 | Key Relevance |
|---|---|---|---|---|---|---|---|
| 1 | Malhotra et al. | LSTM-based encoder-decoder for multi-sensor anomaly detection | ICML Workshop | 2016 | Engines, breathing | — | **Foundational** LSTM-AE for time-series anomaly |
| 2 | Park et al. | A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder | IEEE RA-L | 2018 | Robot sensors | F1≈0.85 | LSTM-VAE; similar architecture motivation |
| 3 | Zhang et al. | A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series | AAAI | 2019 | Air quality, water | AUC 0.93 | Multi-scale approach; early benchmark |
| 4 | Li et al. | MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks | ICANN | 2019 | SWaT, WADI | F1 0.89 | WADI benchmark; GAN-based alternative |
| 5 | Audibert et al. | USAD: Unsupervised Anomaly Detection on Multivariate Time Series | KDD | 2020 | SWaT, WADI, SMD | AUC 0.97 | AE-based; competitive on infrastructure data |

---

## Section 2: Transformer-Based Anomaly Detection

| # | Authors | Title | Venue | Year | Dataset | Key Result | Relevance |
|---|---|---|---|---|---|---|---|
| 6 | Tuli et al. | TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data | VLDB | 2022 | SWaT, MSDS, SMD | F1 0.90 | **Direct baseline** in our comparison |
| 7 | Xu et al. | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy | ICLR | 2022 | SWaT, SMD, MSL | AUC 0.94 | Association discrepancy; competing method |
| 8 | Chen et al. | Learning Graph Structures with Transformer for Multivariate Time Series Anomaly Detection | TNNLS | 2023 | SWaT, WADI | F1 0.91 | Graph + Transformer; not on process data |

---

## Section 3: Process Monitoring — Tennessee Eastman Process (TEP)

| # | Authors | Title | Venue | Year | Method | AUC/DR | Notes |
|---|---|---|---|---|---|---|---|
| 9 | Yin et al. | A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process | J. Process Control | 2012 | PCA, PLS, FDA | F1 <0.7 | Classic comparison; justifies need for DL |
| 10 | Wu & Zhao | Process topology convolutional network model for plant-wide process monitoring | J. Process Control | 2021 | GCN+CNN | AUC 0.92 | Graph-based; uses all 52 TEP variables |
| 11 | Guo et al. | Multiscale Convolutional Neural Networks for Fault Diagnosis of Wind Turbine Gearbox | IEEE TIE | 2019 | Gearbox vibration | Acc 0.98 | Multi-scale CNNs — motivation for our multi-scale |
| 12 | Zhong et al. | Fault Diagnosis for Industrial Processes Using Bidirectional Long Short-Term Memory Recurrent Neural Network | ISA Trans. | 2022 | TEP | F1 0.85 | Bidirectional LSTM; validates BiGRU design |
| 13 | Md Mukhlesur Rahman et al. | Multivariate Industrial Time Series Anomaly Detection with Bidirectional GRU | Expert Systems | 2023 | TEP, NASA | AUC 0.88 | **Very close to our approach** — verify if published |

---

## Section 4: IndPenSim / Bioprocess Monitoring

| # | Authors | Title | Venue | Year | Method | Key Result | Notes |
|---|---|---|---|---|---|---|---|
| 14 | Goldrick et al. | The development of an industrial-scale fed-batch fermentation simulation | J. Biotechnology | 2015 | (IndPenSim paper) | — | **Original IndPenSim** paper — cite this |
| 15 | Goldrick et al. | Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process | Comput. Chem. Eng. | 2019 | MSPC on IndPenSim | F1 <0.55 | Establishes the baseline MSPC result |
| 16 | Rehrl et al. | Process control of a continuous pharmaceutical manufacturing process | Comput. Chem. Eng. | 2020 | IndPenSim, PID | — | Process control context |
| 17 | Tulsyan et al. | Machine-learning for biopharmaceutical batch process monitoring | Comput. Chem. Eng. | 2018 | Simulated batch | — | Nearest to our use case |

---

## Section 5: Early Warning / Fault Prognosis

| # | Authors | Title | Venue | Year | Dataset | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| 18 | Lei et al. | Machinery health prognostics: A systematic review from data acquisition to RUL prediction | Mech. Syst. Signal Process. | 2018 | CMAPSS, bearings | Hours-days | **Survey** — defines prognostics framework |
| 19 | Chen et al. | Remaining Useful Life Prediction of Rotating Machinery Using a Hybrid Deep Learning Framework | Reliab. Eng. Syst. Saf. | 2020 | CMAPSS | RMSE 13.5 | RUL-based; benchmark for prognostics |
| 20 | Zhao et al. | Deep Learning and Its Applications to Machine Health Monitoring | Mech. Syst. Signal Process. | 2019 | Multiple | — | Survey covering LSTM for prognostics |

---

## Section 6: Uncertainty Quantification in Anomaly Detection

| # | Authors | Title | Venue | Year | Method | Notes |
|---|---|---|---|---|---|---|
| 21 | Gal & Ghahramani | Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning | ICML | 2016 | — | **Foundational** MC-Dropout paper |
| 22 | Zhu & Laptev | Deep and Confident Prediction for Time Series at Uber | ICDM Workshop | 2017 | Production TS | — | Uncertainty in TS anomaly detection |
| 23 | Srivastava et al. | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | JMLR | 2014 | Image classif. | — | Original dropout regularization |

---

## Section 7: SHAP / Attribution Methods for Industrial Fault Diagnosis

| # | Authors | Title | Venue | Year | Dataset | Notes |
|---|---|---|---|---|---|---|
| 24 | Lundberg & Lee | A Unified Approach to Interpreting Model Predictions | NeurIPS | 2017 | Various | **Original SHAP** paper |
| 25 | Sundararajan et al. | Axiomatic Attribution for Deep Networks | ICML | 2017 | ImageNet, NLP | **Original IG** paper |
| 26 | Giurgiu & Schumann | Additive Explanations for Anomalies Detected in Multivariate Time Series | ECML-PKDD | 2019 | Industrial sensors | — | SHAP for anomaly in industrial multivariate TS |
| 27 | Tritscher et al. | Evaluation of Post-hoc XAI Approaches Through Synthetic Tabular Data | Data Min. Knowl. Disc. | 2021 | Synthetic | — | XAI evaluation methodology |

---

## Positioning Summary

**What exists in the literature:**
- LSTM/GRU autoencoders for anomaly detection: well-established (refs 1–5)
- TEP benchmarks with DL: several papers achieve AUC 0.88–0.93 (refs 9–13)
- TranAD as direct baseline: established method (ref 6)
- SHAP/IG for industrial systems: limited (ref 26 is closest)
- IndPenSim-specific DL fault detection: **very sparse** (ref 15 shows MSPC baseline)

**Gaps our work addresses:**
1. Leakage-safe evaluation protocol on IndPenSim (not established in prior work)
2. Cross-benchmark comparison (IndPenSim + TEP) with same model
3. Dual XAI (SHAP + IG) on a bioprocess — novelty in domain

**What we CANNOT claim without further evidence:**
- State-of-the-art on TEP (Wu & Zhao 2021 claims AUC 0.92)
- Novel early-warning (the trend alarm for batch 93 is a spurious alarm)
- Significant novelty of BiGRU+Attention architecture (pending ablations)

---

## Recommended Citations for Paper

**Must cite**: Refs 1, 6, 9, 14, 15, 21, 24, 25  
**Should cite**: Refs 3, 10, 12, 18, 26  
**Optional**: Refs 4, 5, 7, 11, 13, 19, 20, 23

> [!NOTE]
> A proper literature review requires you to actually retrieve and read these papers. Several entries above (especially refs 12–13) need DOI verification. Use Google Scholar or IEEE Xplore to confirm. Do not include any citation you cannot verify.
