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


SCORES = np.load(
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

VAL_SCORES = np.load(
    os.path.join(
        RESULTS_DIR,
        "gru_val_scores.npy"
    )
)


THRESHOLD = np.percentile(
    VAL_SCORES,
    99.9
)


def temporal_predictions(
    scores,
    threshold,
    window,
    required
):

    base = (
        scores >
        threshold
    ).astype(int)

    output = np.zeros_like(
        base
    )

    for i in range(
        len(base)
    ):

        start = max(
            0,
            i - window + 1
        )

        count = np.sum(
            base[start:i + 1]
        )

        if count >= required:

            output[i] = 1

    return output


def evaluate(
    predictions
):

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
    print("GRU-AE TEMPORAL FAULT DETECTION")
    print("=" * 70)

    print()

    print(
        "Base threshold:",
        THRESHOLD
    )

    print()

    configurations = [
        (3, 2),
        (5, 2),
        (5, 3),
        (7, 3),
        (7, 4),
        (10, 5)
    ]

    best_f1 = -1
    best_config = None

    for window, required in configurations:

        predictions = temporal_predictions(
            SCORES,
            THRESHOLD,
            window,
            required
        )

        precision, recall, f1, accuracy, fpr = evaluate(
            predictions
        )

        print(
            f"{window:2d} windows | "
            f"{required:2d} required | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"FPR: {fpr:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1

            best_config = (
                window,
                required,
                precision,
                recall,
                f1,
                accuracy,
                fpr
            )

    print()

    print("=" * 70)
    print("BEST TEMPORAL CONFIGURATION")
    print("=" * 70)

    print()

    print(
        "Window:",
        best_config[0]
    )

    print(
        "Required anomalies:",
        best_config[1]
    )

    print(
        "Precision:",
        best_config[2]
    )

    print(
        "Recall:",
        best_config[3]
    )

    print(
        "F1:",
        best_config[4]
    )

    print(
        "Accuracy:",
        best_config[5]
    )

    print(
        "False positive rate:",
        best_config[6]
    )

    print()

    print(
        "Base GRU F1:",
        f"{f1_score(LABELS, (SCORES > THRESHOLD).astype(int)):.4f}"
    )

    print(
        "Temporal GRU F1:",
        f"{best_config[4]:.4f}"
    )

    print()

    print("=" * 70)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()