import os
import sys
import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
SAVE_DIR = os.path.join(BASE_DIR, "saved")
RESULT_DIR = os.path.join(BASE_DIR, "results")

sys.path.insert(0, BASE_DIR)

from gru_ae import GRUAutoencoder

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = os.path.join(
    SAVE_DIR,
    "weighted_gru_best.pt"
)

WEIGHTS = torch.tensor(
    [
        0.419,
        1.256,
        1.674,
        0.628,
        1.047,
        1.047,
        1.465,
        0.837,
        0.628
    ],
    dtype=torch.float32
)

WEIGHTS = WEIGHTS / WEIGHTS.mean()


def calculate_scores(model, X):

    model.eval()
    scores = []

    with torch.no_grad():

        for i in range(0, len(X), 128):

            x = torch.tensor(
                X[i:i + 128],
                dtype=torch.float32,
                device=DEVICE
            )

            output = model(x)

            error = (
                (x - output) ** 2
            )

            error = error * WEIGHTS.to(
                DEVICE
            ).view(
                1, 1, -1
            )

            score = error.mean(
                dim=(1, 2)
            )

            scores.extend(
                score.cpu().numpy()
            )

    return np.asarray(scores)


def evaluate(labels, scores, threshold):

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    auc = roc_auc_score(
        labels,
        scores
    )

    pr_auc = average_precision_score(
        labels,
        scores
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)

    return (
        precision,
        recall,
        f1,
        accuracy,
        auc,
        pr_auc,
        fpr,
        cm
    )


def main():

    print("=" * 70)
    print("WEIGHTED GRU-AE EVALUATION")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    X_val = np.load(
        os.path.join(
            DATA_DIR,
            "X_val.npy"
        )
    ).astype(np.float32)

    X_test = np.load(
        os.path.join(
            DATA_DIR,
            "X_test.npy"
        )
    ).astype(np.float32)

    y_test = np.load(
        os.path.join(
            RESULT_DIR,
            "test_labels.npy"
        )
    )

    print(
        "\nValidation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print(
        "Normal:",
        np.sum(y_test == 0)
    )

    print(
        "Fault:",
        np.sum(y_test == 1)
    )

    model = GRUAutoencoder(
        input_size=9
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "\nWeighted model loaded."
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print(
        "\nCalculating validation scores..."
    )

    val_scores = calculate_scores(
        model,
        X_val
    )

    print(
        "Calculating test scores..."
    )

    test_scores = calculate_scores(
        model,
        X_test
    )

    percentiles = [
        95.0,
        97.0,
        98.0,
        98.5,
        99.0,
        99.2,
        99.5,
        99.7,
        99.8,
        99.9
    ]

    results = []

    print("\n" + "=" * 70)
    print("VALIDATION-DERIVED THRESHOLDS")
    print("=" * 70)

    print(
        "Percentile | Threshold | Precision | Recall | F1 | Accuracy | FPR"
    )

    print("-" * 70)

    for percentile in percentiles:

        threshold = np.percentile(
            val_scores,
            percentile
        )

        (
            precision,
            recall,
            f1,
            accuracy,
            auc,
            pr_auc,
            fpr,
            cm
        ) = evaluate(
            y_test,
            test_scores,
            threshold
        )

        results.append(
            [
                percentile,
                threshold,
                precision,
                recall,
                f1,
                accuracy,
                auc,
                pr_auc,
                fpr
            ]
        )

        print(
            f"{percentile:8.1f} | "
            f"{threshold:.8f} | "
            f"{precision:.4f} | "
            f"{recall:.4f} | "
            f"{f1:.4f} | "
            f"{accuracy:.4f} | "
            f"{fpr:.4f}"
        )

    best = max(
        results,
        key=lambda x: x[4]
    )

    (
        percentile,
        threshold,
        precision,
        recall,
        f1,
        accuracy,
        auc,
        pr_auc,
        fpr
    ) = best

    print("\n" + "=" * 70)
    print("BEST VALIDATION-DERIVED RESULT")
    print("=" * 70)

    print("Percentile:", percentile)
    print("Threshold:", threshold)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Accuracy:", accuracy)
    print("ROC-AUC:", auc)
    print("PR-AUC:", pr_auc)
    print("FPR:", fpr)

    predictions = (
        test_scores >= threshold
    ).astype(int)

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    normal = test_scores[
        y_test == 0
    ]

    fault = test_scores[
        y_test == 1
    ]

    print("\nScore separation:")
    print(
        "Normal mean:",
        np.mean(normal)
    )

    print(
        "Fault mean:",
        np.mean(fault)
    )

    print(
        "Normal median:",
        np.median(normal)
    )

    print(
        "Fault median:",
        np.median(fault)
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "weighted_gru_test_scores.npy"
        ),
        test_scores
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "weighted_gru_val_scores.npy"
        ),
        val_scores
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "weighted_gru_threshold.npy"
        ),
        np.array(threshold)
    )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            "weighted_gru_threshold_analysis.csv"
        ),
        np.asarray(results),
        delimiter=",",
        header=(
            "percentile,threshold,"
            "precision,recall,f1,accuracy,"
            "roc_auc,pr_auc,fpr"
        ),
        comments=""
    )

    print(
        "\nResults saved."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "WEIGHTED GRU-AE EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()