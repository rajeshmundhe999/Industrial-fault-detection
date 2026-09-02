import os
import numpy as np
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

BATCH_IDS = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_batch_ids.npy"
    )
)

TIMES = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_times.npy"
    )
)

VAL_SCORES = np.load(
    os.path.join(
        RESULTS_DIR,
        "val_scores.npy"
    )
)

BASE_THRESHOLD = np.percentile(
    VAL_SCORES,
    99.5
)


def temporal_filter(scores, threshold, window, required):

    raw = (
        scores > threshold
    ).astype(int)

    filtered = np.zeros(
        len(scores),
        dtype=int
    )

    for i in range(
        len(scores)
    ):

        start = max(
            0,
            i - window + 1
        )

        recent = raw[
            start:i + 1
        ]

        if np.sum(recent) >= required:
            filtered[i] = 1

    return filtered


def evaluate(predictions):

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

    fpr = fp / (
        fp + tn
    )

    return (
        precision,
        recall,
        f1,
        accuracy,
        fpr,
        tn,
        fp,
        fn,
        tp
    )


def main():

    print("=" * 70)
    print("TRANAD TEMPORAL FAULT DETECTION")
    print("=" * 70)

    print()

    print(
        "Base threshold:",
        BASE_THRESHOLD
    )

    print()

    configurations = [
        (3, 2),
        (5, 2),
        (5, 3),
        (7, 3),
        (7, 4),
        (10, 5),
    ]

    results = []

    for window, required in configurations:

        predictions = np.zeros(
            len(SCORES),
            dtype=int
        )

        unique_batches = np.unique(
            BATCH_IDS
        )

        for batch in unique_batches:

            mask = (
                BATCH_IDS == batch
            )

            indices = np.where(
                mask
            )[0]

            batch_scores = SCORES[
                indices
            ]

            batch_predictions = temporal_filter(
                batch_scores,
                BASE_THRESHOLD,
                window,
                required
            )

            predictions[
                indices
            ] = batch_predictions

        metrics = evaluate(
            predictions
        )

        (
            precision,
            recall,
            f1,
            accuracy,
            fpr,
            tn,
            fp,
            fn,
            tp
        ) = metrics

        results.append(
            (
                window,
                required,
                precision,
                recall,
                f1,
                accuracy,
                fpr
            )
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

    print()

    print("=" * 70)
    print("BEST TEMPORAL CONFIGURATION")
    print("=" * 70)

    best = max(
        results,
        key=lambda x: x[4]
    )

    (
        window,
        required,
        precision,
        recall,
        f1,
        accuracy,
        fpr
    ) = best

    print()

    print(
        "Window:",
        window
    )

    print(
        "Required anomalies:",
        required
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "F1:",
        f1
    )

    print(
        "Accuracy:",
        accuracy
    )

    print(
        "False positive rate:",
        fpr
    )

    print()

    print("=" * 70)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()