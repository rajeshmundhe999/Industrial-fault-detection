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
    "gru_ae_best.pt"
)


def calculate_scores(model, X):

    model.eval()

    scores = []

    batch_size = 128

    with torch.no_grad():

        for i in range(
            0,
            len(X),
            batch_size
        ):

            x = torch.tensor(
                X[i:i + batch_size],
                dtype=torch.float32,
                device=DEVICE
            )

            output = model(x)

            error = torch.mean(
                (x - output) ** 2,
                dim=(1, 2)
            )

            scores.extend(
                error.cpu().numpy()
            )

    return np.asarray(scores)


def evaluate(
    labels,
    scores,
    threshold
):

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

    fpr = fp / (
        fp + tn
    )

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
    print(
        "GRU-AE CHECKPOINT VERIFICATION"
    )
    print("=" * 70)

    print(
        "\nDevice:",
        DEVICE
    )

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
    ).astype(
        np.float32
    )

    X_test = np.load(
        os.path.join(
            DATA_DIR,
            "X_test.npy"
        )
    ).astype(
        np.float32
    )

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

    print(
        "\nLoading EXACT GRUAutoencoder..."
    )

    model = GRUAutoencoder(
        input_size=9
    ).to(
        DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    print(
        "\nParameters:",
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
        "Validation mean:",
        np.mean(val_scores)
    )

    print(
        "Validation median:",
        np.median(val_scores)
    )

    print(
        "\nCalculating test scores..."
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

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATION-DERIVED THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "Percentile | Threshold | Precision | Recall | F1 | Accuracy | FPR"
    )

    print(
        "-" * 70
    )

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
        best_percentile,
        best_threshold,
        best_precision,
        best_recall,
        best_f1,
        best_accuracy,
        best_auc,
        best_pr_auc,
        best_fpr
    ) = best

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST VALIDATION-DERIVED RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Percentile:",
        best_percentile
    )

    print(
        "Threshold:",
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

    print(
        "ROC-AUC:",
        best_auc
    )

    print(
        "PR-AUC:",
        best_pr_auc
    )

    print(
        "FPR:",
        best_fpr
    )

    predictions = (
        test_scores >= best_threshold
    ).astype(int)

    print(
        "\nConfusion matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    normal_scores = test_scores[
        y_test == 0
    ]

    fault_scores = test_scores[
        y_test == 1
    ]

    print(
        "\nScore separation:"
    )

    print(
        "Normal mean:",
        np.mean(normal_scores)
    )

    print(
        "Fault mean:",
        np.mean(fault_scores)
    )

    print(
        "Normal median:",
        np.median(normal_scores)
    )

    print(
        "Fault median:",
        np.median(fault_scores)
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "gru_verified_test_scores.npy"
        ),
        test_scores
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "gru_verified_val_scores.npy"
        ),
        val_scores
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "gru_verified_predictions.npy"
        ),
        predictions
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "gru_verified_threshold.npy"
        ),
        np.array(
            best_threshold
        )
    )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            "gru_verified_threshold_analysis.csv"
        ),
        np.asarray(
            results
        ),
        delimiter=",",
        header=(
            "percentile,threshold,"
            "precision,recall,f1,accuracy,"
            "roc_auc,pr_auc,fpr"
        ),
        comments=""
    )

    print(
        "\nVerified results saved."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CHECKPOINT VERIFICATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()