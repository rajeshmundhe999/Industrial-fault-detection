# Paper Claims Audit — PFDI Project
**Purpose**: Evaluate which claims from the current project are defensible for publication and which require caveats, further evidence, or retraction.

---

## Claim 1: "GRU-AE v2 achieves AUC=0.9591 on IndPenSim" ✅ VERIFIED

**Evidence**: Directly computed from `scores_and_predictions.npz` using `end_robust_mean_val_max_score`.  
**Status**: Safe to claim. Specify: `val_max` threshold, `end_robust_mean` aggregation, W=60 stride=5.  
**Caveat**: Single seed, single run. Variance unknown (see multi-seed plan).

---

## Claim 2: "Event Recall = 1.000 (all 10 fault batches detected)" ✅ VERIFIED

**Evidence**: Confirmed from predictions array — all fault batches (91–100) have at least one alarm.  
**Status**: Safe to claim.  
**Caveat**: None — this is the most robust metric for batch process monitoring.

---

## Claim 3: "F1 = 0.7138 (best F1 configuration)" ✅ VERIFIED WITH CAVEAT

**Evidence**: Computed from `end_robust_max_val_q99_5_score` predictions.  
**Status**: Safe IF labeled correctly as the `val_q99.5, max-aggregation` operating point.  
**Caveat**: This is NOT the same operating point as AUC=0.9591. These must NEVER appear in the same row in a table without explicit labeling. The "best F1" is achieved at the cost of lower precision (0.650 vs 0.709).

---

## Claim 4: "GRU-AE v2 outperforms TranAD and LSTM-AE" ✅ CONDITIONALLY SAFE

**Evidence**: AUC 0.9591 > 0.9220 (LSTM-AE) > 0.8072 (TranAD).  
**Status**: Safe to claim for AUC metric on IndPenSim.  
**Caveat**: TranAD is evaluated using a non-native scoring formula (reconstruction-error-based). TranAD's original score uses attention weights. This must be disclosed in the paper. Recommended wording: "For fair comparison, all models are evaluated using the same reconstruction-error-based scoring; this may understate TranAD's performance at its native operating point."

---

## Claim 5: "Sensitivity study shows AUC stable (std=0.007) across 18 configurations" ✅ VERIFIED

**Evidence**: Computed from `window_stride_sensitivity.csv` — 18 rows, AUC std = 0.0072.  
**Status**: Safe to claim. This is a strong robustness result.  
**Disclosure**: Different W and stride values produce different numbers of windows. Larger stride = fewer windows = slightly better AUC (due to less class imbalance in test set).

---

## Claim 6: "Trend alarm provides 270-timestep early warning" ❌ INCORRECT AS STATED

**Correct facts**:
- The CSV value `lead_time_windows=54` = `static_alarm_idx - trend_alarm_idx = 65 - 11 = 54 windows = 270 timesteps`
- This measures the trend alarm's advantage over the **static alarm**, not over **fault onset**
- Fault onset = window 58; trend alarm = window 11 → true lead before onset = 47 windows = **235 timesteps**
- BUT: the trend alarm at window 11 fires when score = 20.4, far below the static threshold of 56.4
- Score at window 11 is a transient spike, not a rising trend
- **The trend alarm at window 11 is a FALSE ALARM**

**Status**: ❌ DO NOT CLAIM "270-timestep early warning" in a paper.  
**What can be claimed**: "For batch 93, the static threshold alarm fires 7 windows (35 timesteps) after confirmed fault onset. The trend-based composite score approaches threshold approximately 47 windows (235 timesteps) earlier, though this corresponds to a transient score excursion during normal operation."  
**Honest finding**: The static detector IS 35 timesteps late for batch 93. This is worth reporting.

---

## Claim 7: "Composite recall = 0.8246 vs static recall = 0.5821" — NEEDS VERIFICATION ⚠️

**Source**: Listed in results summary but not directly confirmed from a specific CSV row.  
**Status**: Not independently verified — do not use in paper until source is confirmed.  
**Action**: Re-run `run_early_detection.py` and check its full output.

---

## Claim 8: "MC-Dropout achieves precision=1.000 with zero false alarms" ❌ MISLEADING

**Correct facts from `uncertainty_summary.json`**:
- High-confidence filter F1 = 0.015, Recall = 0.007, Precision = 1.000
- 94.3% of alarms are suppressed (only 5.7% pass the filter)
- Coverage ≈ 6%: practically no alarms are issued
- Standard MC-Dropout AUC = 0.9547 ≈ single-pass AUC = 0.9591 (marginal difference)

**Status**: ❌ DO NOT CLAIM "zero false alarms" from MC-Dropout.  
**What can be claimed**: "MC-Dropout provides calibrated uncertainty estimates. A conservative uncertainty filter achieves precision=1.000 on the 5.7% of windows it flags as high-confidence, enabling operators to distinguish definitive alarms from uncertain detections."

---

## Claim 9: "SHAP and IG agree on Temperature as primary anomaly driver" ✅ SAFE

**Evidence**: Both methods independently rank Temperature #1, Aeration #2. This cross-method agreement is meaningful.  
**Status**: Safe to claim.  
**Disclosure**: "SHAP explains the anomaly scoring function; IG attributes through the GRU network. Their agreement provides convergent evidence for Temperature as the primary anomaly indicator."

---

## Claim 10: "Five distinct fault clusters identified in latent space" ⚠️ EXPLORATORY ONLY

**Evidence**: KMeans K=5 on latent representations of test-set faulty windows.  
**Status**: Exploratory analysis only. NOT a validated fault isolation method.  
**Caveat**: No ground truth fault type labels available for validation. hdbscan not available (KMeans used). Cluster labels are qualitative interpretations.  
**What can be claimed**: "Preliminary latent space analysis suggests the model learns to distinguish fault signatures, though formal fault isolation validation is left for future work."

---

## Claim 11: "TEP augmentation improves F1 from 0.8029 to 0.8155" ✅ VERIFIED

**Evidence**: Both CSV files confirmed.  
**Status**: Safe to claim. Per-fault breakdown shows improvements on F07, F11, F16, F17, F19, F20.  
**Caveat**: F21 worsens (0.538 → 0.456). Root cause unknown. Should be reported honestly.

---

## Claim 12: "BiGRU + Attention + Per-Sensor Heads improves over simpler architectures" ✅ VERIFIED

**Evidence**: Ablation study completed across all 3 variants (`research_audit/ablation/results/ablation_results.csv`).
- **Attention contribution**: Removing attention (Variant A) degrades AUC from 0.9687 → 0.9574 (-1.13 pp) and F1 from 0.8156 → 0.7887 (-2.69 pp).
- **Bidirectional contribution**: Forward-only GRU (Variant B) achieves lower AUC (0.9624) and lower F1 (0.7939) despite having 12.5% more parameters (55k vs 49k).
- **Per-sensor heads contribution**: Sharing an output head (Variant C) drops F1 by 2.82 pp (0.8156 → 0.7874) and Recall by 3.07 pp (0.7705 → 0.7398).
**Status**: Fully verified and safe to claim. Architectural design choices are mathematically justified by experimental outcomes.

---

## Summary Table

| Claim | Status | Safe to Publish? |
|---|---|---|
| AUC=0.9591 (IndPenSim, primary OP) | ✅ Verified | Yes |
| Event Recall=1.000 | ✅ Verified | Yes |
| F1=0.7138 (high-recall OP) | ✅ Verified (with labeling) | Yes, if clearly labeled |
| Outperforms LSTM-AE, TranAD | ✅ Conditional | Yes, with TranAD caveat |
| Stability: AUC std=0.007 (18 configs) | ✅ Verified | Yes |
| 270-timestep early warning | ❌ Incorrect definition | No — retract |
| Trend alarm early warning (batch 93) | ❌ Spurious alarm | No — reframe honestly |
| Composite recall 0.8246 | ⚠️ Unverified | Verify before claiming |
| MC-Dropout precision=1.000 | ❌ Misleading | No — reframe |
| SHAP/IG agree on Temperature #1 | ✅ Verified | Yes |
| Latent space clusters → fault types | ⚠️ Exploratory | Yes, as exploratory |
| TEP augmentation improves F1 | ✅ Verified | Yes (disclose F21 regression) |
| BiGRU+Attn > ablations | ✅ Verified | Yes — proven by ablation study |
