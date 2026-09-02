import os
import sys
import numpy as np
import torch
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from gru_ae import GRUAutoencoder


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


SENSORS = [
    "Agitator RPM",
    "Aeration rate",
    "Sugar feed rate",
    "Acid flow rate",
    "Base flow rate",
    "Heating/cooling water flow",
    "Dissolved oxygen",
    "pH",
    "Temperature"
]


def load_model(device):

    model = GRUAutoencoder(
        input_size=9
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def calculate_errors(
    model,
    X,
    device,
    batch_size=256
):

    errors = []

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
                dtype=torch.float32,
                device=device
            )

            reconstruction = model(x)

            error = (
                x - reconstruction
            ) ** 2

            error = error.mean(
                dim=1
            )

            errors.append(
                error.cpu().numpy()
            )

    return np.concatenate(
        errors,
        axis=0
    )


def robust_normalize(
    val_scores,
    test_scores
):

    median = np.median(
        val_scores,
        axis=0
    )

    q95 = np.percentile(
        val_scores,
        95,
        axis=0
    )

    scale = q95 - median

    scale[
        scale < 1e-12
    ] = 1e-12

    val_normalized = (
        val_scores - median
    ) / scale

    test_normalized = (
        test_scores - median
    ) / scale

    return (
        val_normalized,
        test_normalized
    )


def evaluate(
    y,
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    accuracy = np.mean(
        predictions == y
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (
        fp + tn
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr
    }


def main():

    print("=" * 75)
    print(
        "VALIDATION-BASED SENSOR WEIGHT SEARCH"
    )
    print("=" * 75)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device
    )

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

    y_val = np.load(
        os.path.join(
            DATA_DIR,
            "y_val.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            DATA_DIR,
            "y_test.npy"
        )
    )

    print()
    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print()
    print(
        "Validation labels:",
        np.bincount(y_val)
    )

    print(
        "Test labels:",
        np.bincount(y_test)
    )

    print()
    print("Loading model...")

    model = load_model(
        device
    )

    print("Model loaded.")

    print()
    print(
        "Calculating validation reconstruction errors..."
    )

    val_errors = calculate_errors(
        model,
        X_val,
        device
    )

    print(
        "Calculating test reconstruction errors..."
    )

    test_errors = calculate_errors(
        model,
        X_test,
        device
    )

    val_errors, test_errors = robust_normalize(
        val_errors,
        test_errors
    )

    print()
    print("=" * 75)
    print(
        "SEARCHING VALIDATION-OPTIMAL WEIGHTS"
    )
    print("=" * 75)

    candidates = []

    sensor_pairs = [
        (1, 2),
        (1, 5),
        (2, 5),
        (1, 2, 5),
        (1, 2, 4, 5),
        (1, 2, 4, 5, 6),
        (1, 2, 3, 4, 5, 6, 7, 8)
    ]

    for sensors in sensor_pairs:

        if len(sensors) == 2:

            for sugar_weight in np.arange(
                0.0,
                1.01,
                0.05
            ):

                weights = np.array([
                    sugar_weight,
                    1.0 - sugar_weight
                ])

                val_score = (
                    val_errors[
                        :,
                        list(sensors)
                    ] * weights
                ).sum(
                    axis=1
                )

                best_threshold = None

                for percentile in np.arange(
                    80,
                    99.91,
                    0.1
                ):

                    threshold = np.percentile(
                        val_score,
                        percentile
                    )

                    prediction = (
                        val_score >= threshold
                    ).astype(int)

                    f1 = f1_score(
                        y_val,
                        prediction,
                        zero_division=0
                    )

                    if (
                        best_threshold is None
                        or f1 >
                        best_threshold["f1"]
                    ):

                        best_threshold = {
                            "sensors": sensors,
                            "weights": weights.copy(),
                            "percentile": percentile,
                            "threshold": threshold,
                            "f1": f1
                        }

                candidates.append(
                    best_threshold
                )

        else:

            weights = np.ones(
                len(sensors)
            )

            weights = (
                weights /
                weights.sum()
            )

            val_score = (
                val_errors[
                    :,
                    list(sensors)
                ] * weights
            ).sum(
                axis=1
            )

            best_threshold = None

            for percentile in np.arange(
                80,
                99.91,
                0.1
            ):

                threshold = np.percentile(
                    val_score,
                    percentile
                )

                prediction = (
                    val_score >= threshold
                ).astype(int)

                f1 = f1_score(
                    y_val,
                    prediction,
                    zero_division=0
                )

                if (
                    best_threshold is None
                    or f1 >
                    best_threshold["f1"]
                ):

                    best_threshold = {
                        "sensors": sensors,
                        "weights": weights,
                        "percentile": percentile,
                        "threshold": threshold,
                        "f1": f1
                    }

            candidates.append(
                best_threshold
            )

    candidates = sorted(
        candidates,
        key=lambda x: x["f1"],
        reverse=True
    )

    print()

    for i, result in enumerate(
        candidates[:15],
        1
    ):

        names = [
            SENSORS[j]
            for j in result["sensors"]
        ]

        print(
            f"{i:2d}. "
            f"F1={result['f1']:.4f} | "
            f"Percentile="
            f"{result['percentile']:.2f} | "
            f"Sensors={names} | "
            f"Weights="
            f"{result['weights']}"
        )

    print()
    print("=" * 75)
    print(
        "FINAL TEST EVALUATION"
    )
    print("=" * 75)

    final_results = []

    for result in candidates[:15]:

        sensors = list(
            result["sensors"]
        )

        weights = result[
            "weights"
        ]

        threshold = result[
            "threshold"
        ]

        test_score = (
            test_errors[
                :,
                sensors
            ] * weights
        ).sum(
            axis=1
        )

        metrics = evaluate(
            y_test,
            test_score,
            threshold
        )

        auc = roc_auc_score(
            y_test,
            test_score
        )

        pr_auc = average_precision_score(
            y_test,
            test_score
        )

        final_results.append({
            "sensors": ",".join(
                SENSORS[j]
                for j in sensors
            ),
            "weights": ",".join(
                f"{w:.3f}"
                for w in weights
            ),
            "validation_f1":
                result["f1"],
            "percentile":
                result["percentile"],
            "threshold":
                threshold,
            "test_precision":
                metrics["precision"],
            "test_recall":
                metrics["recall"],
            "test_f1":
                metrics["f1"],
            "test_accuracy":
                metrics["accuracy"],
            "test_fpr":
                metrics["fpr"],
            "test_auc":
                auc,
            "test_pr_auc":
                pr_auc
        })

    final_results = sorted(
        final_results,
        key=lambda x: x["test_f1"],
        reverse=True
    )

    print()

    for i, result in enumerate(
        final_results,
        1
    ):

        print(
            f"{i:2d}. "
            f"Test F1="
            f"{result['test_f1']:.4f} | "
            f"Precision="
            f"{result['test_precision']:.4f} | "
            f"Recall="
            f"{result['test_recall']:.4f} | "
            f"AUC="
            f"{result['test_auc']:.4f}"
        )

        print(
            f"    Sensors: "
            f"{result['sensors']}"
        )

        print(
            f"    Weights: "
            f"{result['weights']}"
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    output = os.path.join(
        RESULTS_DIR,
        "validation_weight_search.csv"
    )

    pd.DataFrame(
        final_results
    ).to_csv(
        output,
        index=False
    )

    print()
    print(
        "Results saved:"
    )

    print(output)

    print()
    print("=" * 75)
    print(
        "VALIDATION WEIGHT SEARCH COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()