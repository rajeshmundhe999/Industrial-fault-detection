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
    confusion_matrix
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.gru_ae import GRUAutoencoder


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "saved",
    "gru_ae_best.pt"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def calculate_scores(
    model,
    X
):

    scores = []

    batch_size = 512

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(X)
            )

            x = torch.tensor(
                X[start:end],
                dtype=torch.float32
            ).to(
                DEVICE
            )

            reconstruction = model(
                x
            )

            error = torch.mean(
                (
                    reconstruction -
                    x
                ) ** 2,
                dim=(1, 2)
            )

            scores.extend(
                error.cpu().numpy()
            )

    return np.asarray(
        scores
    )


def main():

    print("=" * 60)
    print("GRU-AE ANOMALY DETECTION EVALUATION")
    print("=" * 60)

    print()

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    X_val = np.load(
        os.path.join(
            DATA_DIR,
            "X_val.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            DATA_DIR,
            "X_test.npy"
        )
    )

    labels = np.load(
        os.path.join(
            RESULTS_DIR,
            "test_labels.npy"
        )
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print(
        "Normal test windows:",
        np.sum(labels == 0)
    )

    print(
        "Fault test windows:",
        np.sum(labels == 1)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = GRUAutoencoder(
        input_size=9
    ).to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()

    print(
        "Model loaded."
    )

    print()

    print(
        "Calculating validation scores..."
    )

    val_scores = calculate_scores(
        model,
        X_val
    )

    threshold = np.percentile(
        val_scores,
        99.5
    )

    print(
        "Validation 99.5th percentile:",
        threshold
    )

    print()

    print(
        "Calculating test scores..."
    )

    test_scores = calculate_scores(
        model,
        X_test
    )

    predictions = (
        test_scores >
        threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

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

    auc = roc_auc_score(
        labels,
        test_scores
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions
    ).ravel()

    fpr = fp / (
        fp + tn
    )

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print()

    print(
        "Threshold:",
        threshold
    )

    print(
        "Accuracy:",
        f"{accuracy:.4f}"
    )

    print(
        "Precision:",
        f"{precision:.4f}"
    )

    print(
        "Recall:",
        f"{recall:.4f}"
    )

    print(
        "F1:",
        f"{f1:.4f}"
    )

    print(
        "ROC-AUC:",
        f"{auc:.4f}"
    )

    print(
        "False positive rate:",
        f"{fpr:.4f}"
    )

    print()

    print(
        "Confusion matrix:"
    )

    print(
        np.array(
            [
                [tn, fp],
                [fn, tp]
            ]
        )
    )

    print()

    print(
        "Normal mean:",
        np.mean(
            test_scores[
                labels == 0
            ]
        )
    )

    print(
        "Fault mean:",
        np.mean(
            test_scores[
                labels == 1
            ]
        )
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "gru_test_scores.npy"
        ),
        test_scores
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "gru_val_scores.npy"
        ),
        val_scores
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "gru_threshold.npy"
        ),
        np.array(
            threshold
        )
    )

    print()

    print(
        "Results saved."
    )

    print()
    print("=" * 60)
    print("GRU-AE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()