import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from tqdm import tqdm

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.tranad import TranAD


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

MODEL_FILE = os.path.join(
    BASE_DIR,
    "model",
    "saved",
    "tranad_best.pt"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)

BATCH_SIZE = 128

THRESHOLD_PERCENTILE = 99.0


def load_data():

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

    y_test = np.load(
        os.path.join(
            DATA_DIR,
            "y_test.npy"
        )
    )

    batch_ids = np.load(
        os.path.join(
            DATA_DIR,
            "batch_ids_test.npy"
        )
    )

    times = np.load(
        os.path.join(
            DATA_DIR,
            "times_test.npy"
        )
    )

    return (
        X_val,
        X_test,
        y_test,
        batch_ids,
        times
    )


def load_model(device):

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model = TranAD(
        n_features=checkpoint["n_features"],
        window_size=checkpoint["window_size"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def calculate_scores(
    model,
    data,
    device
):

    scores = []

    with torch.no_grad():

        for start in tqdm(
            range(
                0,
                len(data),
                BATCH_SIZE
            ),
            desc="Calculating anomaly scores"
        ):

            end = min(
                start + BATCH_SIZE,
                len(data)
            )

            batch = torch.tensor(
                data[start:end],
                dtype=torch.float32,
                device=device
            )

            reconstruction_1, reconstruction_2 = (
                model(batch)
            )

            error_1 = torch.mean(
                (
                    batch - reconstruction_1
                ) ** 2,
                dim=(1, 2)
            )

            error_2 = torch.mean(
                (
                    batch - reconstruction_2
                ) ** 2,
                dim=(1, 2)
            )

            score = (
                0.5 * error_1 +
                0.5 * error_2
            )

            scores.append(
                score.cpu().numpy()
            )

    return np.concatenate(
        scores
    )


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("TRANAD ANOMALY DETECTION EVALUATION")
    print("=" * 60)

    print()

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    (
        X_val,
        X_test,
        y_test,
        batch_ids,
        times
    ) = load_data()

    print()

    print(
        "Validation shape:",
        X_val.shape
    )

    print(
        "Test shape:",
        X_test.shape
    )

    model = load_model(
        device
    )

    print()
    print("Calculating validation scores...")

    val_scores = calculate_scores(
        model,
        X_val,
        device
    )

    threshold = np.percentile(
        val_scores,
        THRESHOLD_PERCENTILE
    )

    print()

    print(
        f"Validation {THRESHOLD_PERCENTILE}th percentile:"
        f" {threshold:.8f}"
    )

    print()
    print("Calculating test scores...")

    test_scores = calculate_scores(
        model,
        X_test,
        device
    )

    predictions = (
        test_scores >= threshold
    ).astype(
        np.int64
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
        "Actual normal windows:",
        np.sum(y_test == 0)
    )

    print(
        "Actual fault windows:",
        np.sum(y_test == 1)
    )

    print(
        "Predicted normal windows:",
        np.sum(predictions == 0)
    )

    print(
        "Predicted anomaly windows:",
        np.sum(predictions == 1)
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        test_scores
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print()

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {auc:.4f}"
    )

    print()

    print("Confusion matrix:")

    print(cm)

    print()

    print("Score statistics:")

    print(
        "Normal mean:",
        np.mean(
            test_scores[y_test == 0]
        )
    )

    print(
        "Normal median:",
        np.median(
            test_scores[y_test == 0]
        )
    )

    print(
        "Fault mean:",
        np.mean(
            test_scores[y_test == 1]
        )
    )

    print(
        "Fault median:",
        np.median(
            test_scores[y_test == 1]
        )
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "val_scores.npy"
        ),
        val_scores
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "test_scores.npy"
        ),
        test_scores
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "test_predictions.npy"
        ),
        predictions
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "test_labels.npy"
        ),
        y_test
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "test_batch_ids.npy"
        ),
        batch_ids
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "test_times.npy"
        ),
        times
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "threshold.npy"
        ),
        np.array(
            threshold
        )
    )

    print()

    print(
        "Results saved to:",
        RESULTS_DIR
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()