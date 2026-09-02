import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "model",
    "plots"
)

SCORES = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_scores.npy"
    )
)

LABELS = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_labels.npy"
    )
)

VAL_SCORES = np.load(
    os.path.join(
        RESULTS_DIR,
        "val_scores.npy"
    )
)

print("=" * 60)
print("TRANAD THRESHOLD ANALYSIS")
print("=" * 60)

print()

print("Test windows:", len(SCORES))
print("Normal:", np.sum(LABELS == 0))
print("Fault:", np.sum(LABELS == 1))

print()


percentiles = [
    99.0,
    99.2,
    99.5,
    99.7,
    99.9
]

thresholds = []

for p in percentiles:

    threshold = np.percentile(
        VAL_SCORES,
        p
    )

    thresholds.append(
        threshold
    )


print("Validation-derived thresholds")
print("-" * 60)

for p, threshold in zip(
    percentiles,
    thresholds
):

    predictions = (
        SCORES > threshold
    ).astype(int)

    precision = precision_score(
        LABELS,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        LABELS,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        LABELS,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        LABELS,
        predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        LABELS,
        predictions
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
    )

    print(
        f"{p:5.1f}% | "
        f"Threshold: {threshold:.8f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"FPR: {false_positive_rate:.4f}"
    )


print()
print("=" * 60)
print("DIAGNOSTIC TEST-OPTIMAL THRESHOLD")
print("=" * 60)

candidate_thresholds = np.linspace(
    np.min(SCORES),
    np.max(SCORES),
    500
)

best_f1 = -1
best_threshold = None
best_precision = None
best_recall = None
best_accuracy = None

f1_values = []
precision_values = []
recall_values = []

for threshold in candidate_thresholds:

    predictions = (
        SCORES > threshold
    ).astype(int)

    precision = precision_score(
        LABELS,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        LABELS,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        LABELS,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        LABELS,
        predictions
    )

    precision_values.append(
        precision
    )

    recall_values.append(
        recall
    )

    f1_values.append(
        f1
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold
        best_precision = precision
        best_recall = recall
        best_accuracy = accuracy


print(
    "Best threshold:",
    best_threshold
)

print(
    "Precision:",
    best_precision
)

print(
    "Recall:",
    best_recall
)

print(
    "F1:",
    best_f1
)

print(
    "Accuracy:",
    best_accuracy
)

print()

print(
    "NOTE: This threshold is diagnostic only."
)

print(
    "It was optimized using the labeled test set."
)

print(
    "It must NOT be reported as an unbiased final threshold."
)


os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    candidate_thresholds,
    precision_values,
    label="Precision"
)

plt.plot(
    candidate_thresholds,
    recall_values,
    label="Recall"
)

plt.plot(
    candidate_thresholds,
    f1_values,
    label="F1"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label="Best F1 threshold"
)

plt.xlabel(
    "Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "TranAD Threshold Trade-off"
)

plt.legend()

plt.tight_layout()

output = os.path.join(
    PLOT_DIR,
    "threshold_tradeoff.png"
)

plt.savefig(
    output,
    dpi=200
)

plt.close()

print()
print(
    "Saved:",
    output
)

print()
print("=" * 60)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 60)