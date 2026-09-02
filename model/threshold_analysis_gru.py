import os
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
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


VAL_SCORES = np.load(
    os.path.join(
        RESULTS_DIR,
        "gru_val_scores.npy"
    )
)

TEST_SCORES = np.load(
    os.path.join(
        RESULTS_DIR,
        "gru_test_scores.npy"
    )
)

LABELS = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_labels.npy"
    )
)


def evaluate_threshold(threshold):

    predictions = (
        TEST_SCORES >
        threshold
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

    tn = np.sum(
        (LABELS == 0) &
        (predictions == 0)
    )

    fp = np.sum(
        (LABELS == 0) &
        (predictions == 1)
    )

    fpr = fp / (
        tn + fp
    )

    return (
        precision,
        recall,
        f1,
        accuracy,
        fpr
    )


def main():

    print("=" * 70)
    print("GRU-AE THRESHOLD ANALYSIS")
    print("=" * 70)

    print()

    print(
        "Test windows:",
        len(TEST_SCORES)
    )

    print(
        "Normal:",
        np.sum(LABELS == 0)
    )

    print(
        "Fault:",
        np.sum(LABELS == 1)
    )

    print()

    percentiles = [
        98.0,
        98.5,
        99.0,
        99.2,
        99.5,
        99.7,
        99.8,
        99.9
    ]

    print(
        "Validation-derived thresholds"
    )

    print("-" * 70)

    best_f1 = -1
    best_info = None

    for percentile in percentiles:

        threshold = np.percentile(
            VAL_SCORES,
            percentile
        )

        precision, recall, f1, accuracy, fpr = (
            evaluate_threshold(
                threshold
            )
        )

        print(
            f"{percentile:5.1f}% | "
            f"Threshold: {threshold:.8f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"FPR: {fpr:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1

            best_info = (
                percentile,
                threshold,
                precision,
                recall,
                f1,
                accuracy,
                fpr
            )

    print()

    print("=" * 70)
    print("BEST VALIDATION-DERIVED OPERATING POINT")
    print("=" * 70)

    print()

    print(
        "Best percentile:",
        best_info[0]
    )

    print(
        "Threshold:",
        best_info[1]
    )

    print(
        "Precision:",
        best_info[2]
    )

    print(
        "Recall:",
        best_info[3]
    )

    print(
        "F1:",
        best_info[4]
    )

    print(
        "Accuracy:",
        best_info[5]
    )

    print(
        "FPR:",
        best_info[6]
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Threshold selection uses validation data only."
    )

    print(
        "Test labels are used only for final evaluation."
    )

    print()

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()