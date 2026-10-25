# PFDI Project — Research Audit Index
**Last Updated**: 2026-10-19  
**Project**: BCSE497J Process Fault Detection & Isolation

---

## Quick Status

| Item | Status |
|---|---|
| Repository Audit | ✅ Complete |
| Results Verification | ✅ Complete |
| Literature Review | ✅ Complete (needs DOI verification) |
| Experiment Plan | ✅ Complete |
| Reproducibility Guide | ✅ Complete |
| Paper Claims Audit | ✅ Complete |
| Paper Draft | ✅ Written (ablation section pending) |
| Architecture Ablation | ✅ Complete (All 3 variants evaluated) |
| Streamlit Dashboard | ✅ Complete — `py -3.12 -m streamlit run dashboard/app.py` |
| Batch 93 Investigation | ✅ Figures generated |
| Multi-seed reproducibility | ❌ Not yet run |

---

## Key Findings

### ✅ Confirmed Results
- **Architecture Ablations (All 3 design choices validated)**:
  - Full v2: AUC = **0.9687**, F1 = **0.8156**, Recall = **0.7705**, Event Recall = **1.000**
  - Variant A (No Attention): AUC = 0.9574 (-1.13 pp), F1 = 0.7887 (-2.69 pp)
  - Variant B (Unidirectional): AUC = 0.9624 (-0.63 pp), F1 = 0.7939 (-2.17 pp)
  - Variant C (Shared Head): AUC = 0.9585 (-1.02 pp), F1 = 0.7874 (-2.82 pp)
- **AUC = 0.9591** (IndPenSim, primary OP: val_max threshold, mean aggregation)
- **Event Recall = 1.000** for all 10 fault batches
- **AUC std = 0.007** across 18 sensitivity configs
- **SHAP and IG agree**: Temperature #1, Aeration #2
- **TEP AUC**: 0.8721 (original) → 0.8754 (augmented)

### ❌ Claims Requiring Correction
- **"270-timestep early warning"**: The trend alarm for Batch 93 fires at window 11 with score=20.4 (threshold=56.4). This is a **transient spike during normal operation**, not early warning. True finding: static alarm fires 35 timesteps AFTER fault onset for Batch 93.
- **"MC-Dropout precision=1.000"**: True, but coverage = 5.7% (only 2 alarms). Not meaningful for publication.

### ⚠️ Pending
- Multi-seed results (AUC mean ± std over 3 seeds)

---

## Files in This Directory

| File | Purpose |
|---|---|
| [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md) | Project structure, what exists, versions |
| [RESULTS_AUDIT.md](RESULTS_AUDIT.md) | Full metric verification with sources |
| [PAPER_CLAIMS_AUDIT.md](PAPER_CLAIMS_AUDIT.md) | Which claims are safe/unsafe for publication |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | Priority-ranked remaining experiments |
| [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) | ~27 relevant papers across 7 topic areas |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Step-by-step reproduction commands |
| [PAPER_DRAFT.md](PAPER_DRAFT.md) | Full paper draft (ablation section TBD) |
| [ablation/train_ablation.py](ablation/train_ablation.py) | Architecture ablation training script |
| [plot_batch93.py](plot_batch93.py) | Batch 93 score curve investigation |
| [figures/](figures/) | Generated plots |

---

## What to Do Next

### Immediate (Today)
1. **Wait for ablation to finish** (~9 hrs): `research_audit/ablation/ablation_training.log`
2. **Run dashboard**: Double-click `START_DASHBOARD.bat` or run `py -3.12 -m streamlit run dashboard/app.py`

### This Week
3. **Fill ablation results** into `PAPER_DRAFT.md` Section 4.4
4. **Verify literature citations** via Google Scholar (see LITERATURE_REVIEW.md)
5. **Run multi-seed** for statistical confidence

### Next Week
6. **Submit paper draft** to supervisor for feedback
7. **Format for target venue** (IEEE Access or Expert Systems with Applications)
