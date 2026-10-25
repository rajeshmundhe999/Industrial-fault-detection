# Experiment Plan — PFDI Project
**Date**: 2026-10-19  
**Status**: Pending execution

---

## Research Question

> "Can a reconstruction-based anomaly detector with trend-aware alarm logic provide earlier and more operationally useful fault warnings than a static reconstruction-based detector, without an unacceptable increase in nuisance alarms?"

---

## Priority Ranking

| Priority | Experiment | Value | Effort |
|---|---|---|---|
| P1 | Architecture ablation (3 variants) | Validates BiGRU+Attn contribution | 3×3 hrs CPU |
| P2 | Multi-seed reproducibility (3 seeds) | Required for statistical claims | 3×3 hrs CPU |
| P3 | Corrected early-warning evaluation at fixed FPR | Core contribution validation | 2 hrs |
| P4 | Batch 93 score curve investigation | Clarify false-alarm vs early detection | 30 min |
| P5 | Streamlit dashboard | College demo | 4 hrs |
| P6 | Paper draft | Publication goal | 8 hrs |

---

## P1: Architecture Ablation

**Goal**: Quantitatively show that the BiGRU+Attention+per-sensor-heads design beats simpler alternatives. If it doesn't, the architecture contribution claim is invalid.

**Variants to test:**

| Variant | Change from v2 | Expected effect |
|---|---|---|
| A: No Attention | Remove attention pooling, use last hidden state | Should hurt — attention selects informative timesteps |
| B: Unidirectional GRU | Replace BiGRU with forward-only GRU | Should hurt moderately |
| C: Shared head | Replace per-sensor heads with single linear head | Should hurt — sensors have different normal ranges |
| D: Full (v2) | Current model | Baseline |

**Script**: `research_audit/ablation/train_ablation.py`  
**Results**: `research_audit/ablation/results/`

**Decision rule**: If BiGRU+Attn v2 doesn't achieve ≥1pp AUC improvement over ALL three ablations, we cannot claim the architecture as a contribution. We can still claim the evaluation protocol and early-warning mechanism.

---

## P2: Multi-Seed Reproducibility

**Goal**: Report mean ± std over 3 seeds for the primary GRU-AE v2 model.

**Seeds**: 42, 7, 123  
**Metric**: AUC (primary), F1, Event Recall

**Script**: `research_audit/multiseed/train_multiseed.py`  
**Expected output**: `AUC = 0.959 ± 0.003` (hypothetical)

---

## P3: Corrected Early-Warning Evaluation

**Current problem**: "lead_time" = static_alarm - trend_alarm (not fault onset)

**Correct evaluation protocol**:
1. For each faulty batch, find `fault_onset_window` = first window labeled faulty
2. For each alarm method, find `first_alarm_window` = first window where score > threshold
3. `true_lead_time = fault_onset_window - first_alarm_window` (positive = early, negative = late)
4. Report at **fixed FPR operating points** (e.g., FPR=0.05, FPR=0.10)
5. Compare: static alarm vs trend-composite alarm vs MC-Dropout alarm

**Key question**: Does the trend alarm reliably detect faults before onset, or does it fire during normal operation (false alarm)?

**Script**: `research_audit/eval_earlywarning.py`

---

## P4: Batch 93 Score Curve Investigation

**Goal**: Plot the GRU-AE v2 anomaly score curve for batch 93 over all 199 windows. Verify:
- Is score rising from window 11 onward, or is it a spike?
- Does the rising score precede fault onset (window 58)?
- Or does trend alarm at window 11 fire during a period of stable normal scores (spurious alarm)?

**Script**: `research_audit/plot_batch93.py`  
**Output**: `research_audit/figures/batch93_score_curve.png`

---

## P5: Streamlit Dashboard

**Features required for college demo:**
1. Landing page: project title, team, supervisor
2. IndPenSim tab: per-batch score plots, comparison table
3. TEP tab: per-fault detection rates bar chart
4. Early Warning tab: batch 93 score curve with alarm annotations
5. XAI tab: SHAP bar plot, sensor attribution table
6. Sensitivity tab: heatmap of AUC vs window × stride

**Script**: `dashboard/app.py`

---

## P6: Paper Draft Structure

Target venue: **IEEE Transactions on Industrial Informatics** (SCI Q1) or  
**Expert Systems with Applications** (SCI Q1, more accessible)

**Outline:**
1. Introduction — process fault detection importance, limitation of static thresholds
2. Related Work — GRU-AE methods, TranAD, early-warning approaches
3. Methodology — data protocol, GRU-AE v2 architecture, scoring policies, trend alarm
4. Experimental Setup — IndPenSim, TEP, leakage-free protocol, baselines
5. Results — comparison table, sensitivity, TEP per-fault, early warning
6. Discussion — honest limitations, what the trend alarm actually does
7. Conclusion

**DO NOT claim**:
- "State-of-the-art" without literature comparison
- "Zero false alarms" from MC-Dropout high-confidence filter
- "270 timestep lead" without correct definition

---

## Timeline

| Week | Tasks |
|---|---|
| Week 1 | P1 ablations (train), P4 investigation, Dashboard skeleton |
| Week 2 | P2 multi-seed, P3 corrected early-warning, Literature review |
| Week 3 | Dashboard complete, Paper draft |
| Week 4 | Revisions, Finalization |
