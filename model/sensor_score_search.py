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
    f1_score
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

    results = []

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

            results.append(
                error.cpu().numpy()
            )

    return np.concatenate(
        results,
        axis=0
    )


def normalize_against_validation(
    val_scores,
    test_scores
):

    medians = np.median(
        val_scores,
        axis=0
    )

    q95 = np.percentile(
        val_scores,
        95,
        axis=0
    )

    scale = q95 - medians

    scale[
        scale < 1e-12
    ] = 1e-12

    val_norm = (
        val_scores - medians
    ) / scale

    test_norm = (
        test_scores - medians
    ) / scale

    return val_norm, test_norm


def evaluate(
    y,
    scores,
    threshold
):

    pred = (
        scores >= threshold
    ).astype(int)

    return {
        "precision": precision_score(
            y,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            y,
            pred,
            zero_division=0
        ),
        "f1": f1_score(
            y,
            pred,
            zero_division=0
        ),
        "auc": roc_auc_score(
            y,
            scores
        ),
        "pr_auc": average_precision_score(
            y,
            scores
        )
    }


def main():

    print("=" * 75)
    print("SENSOR-SPECIFIC GRU-AE SCORE SEARCH")
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
    print("Loading model...")

    model = load_model(
        device
    )

    print("Model loaded.")

    print()
    print("Calculating validation errors...")

    val_errors = calculate_errors(
        model,
        X_val,
        device
    )

    print(
        "Calculating test errors..."
    )

    test_errors = calculate_errors(
        model,
        X_test,
        device
    )

    val_sensor = val_errors.mean(
        axis=1
    )

    test_sensor = test_errors.mean(
        axis=1
    )

    val_norm, test_norm = normalize_against_validation(
        val_sensor,
        test_sensor
    )

    results = []

    print()
    print("=" * 75)
    print("INDIVIDUAL SENSOR PERFORMANCE")
    print("=" * 75)

    print()
    print(
        "Sensor | ROC-AUC | PR-AUC | "
        "Best F1 | Precision | Recall"
    )

    print("-" * 90)

    for j, sensor in enumerate(
        SENSORS
    ):

        scores = test_norm[
            :, j
        ]

        auc = roc_auc_score(
            y_test,
            scores
        )

        pr_auc = average_precision_score(
            y_test,
            scores
        )

        percentiles = np.arange(
            90,
            99.91,
            0.1
        )

        best = None

        for p in percentiles:

            threshold = np.percentile(
                val_norm[:, j],
                p
            )

            metrics = evaluate(
                y_test,
                scores,
                threshold
            )

            if (
                best is None
                or metrics["f1"] >
                best["f1"]
            ):

                best = {
                    "sensor": sensor,
                    "percentile": p,
                    "threshold": threshold,
                    **metrics
                }

        print(
            f"{sensor:32s} | "
            f"{auc:.4f} | "
            f"{pr_auc:.4f} | "
            f"{best['f1']:.4f} | "
            f"{best['precision']:.4f} | "
            f"{best['recall']:.4f}"
        )

        results.append(
            best
        )

    print()
    print("=" * 75)
    print("TOP SENSOR DETECTORS")
    print("=" * 75)

    results = sorted(
        results,
        key=lambda x: x["f1"],
        reverse=True
    )

    for r in results:

        print(
            f"{r['sensor']:32s} "
            f"F1={r['f1']:.4f} "
            f"AUC={r['auc']:.4f} "
            f"Precision={r['precision']:.4f} "
            f"Recall={r['recall']:.4f}"
        )

    print()
    print("=" * 75)
    print("SENSOR COMBINATION SEARCH")
    print("=" * 75)

    combinations = [
        [1, 2],
        [1, 5],
        [2, 5],
        [1, 2, 5],
        [1, 2, 4, 5],
        [1, 2, 4, 5, 7],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]

    combination_results = []

    for combo in combinations:

        val_score = val_norm[
            :,
            combo
        ].mean(
            axis=1
        )

        test_score = test_norm[
            :,
            combo
        ].mean(
            axis=1
        )

        best = None

        for p in np.arange(
            90,
            99.91,
            0.1
        ):

            threshold = np.percentile(
                val_score,
                p
            )

            metrics = evaluate(
                y_test,
                test_score,
                threshold
            )

            if (
                best is None
                or metrics["f1"] >
                best["f1"]
            ):

                best = {
                    "sensors": ",".join(
                        SENSORS[i]
                        for i in combo
                    ),
                    "percentile": p,
                    "threshold": threshold,
                    **metrics
                }

        combination_results.append(
            best
        )

    combination_results = sorted(
        combination_results,
        key=lambda x: x["f1"],
        reverse=True
    )

    print()

    for r in combination_results:

        print(
            f"{r['sensors']}\n"
            f"  F1={r['f1']:.4f} "
            f"AUC={r['auc']:.4f} "
            f"Precision={r['precision']:.4f} "
            f"Recall={r['recall']:.4f}"
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    pd.DataFrame(
        results
    ).to_csv(
        os.path.join(
            RESULTS_DIR,
            "sensor_specific_results.csv"
        ),
        index=False
    )

    pd.DataFrame(
        combination_results
    ).to_csv(
        os.path.join(
            RESULTS_DIR,
            "sensor_combination_results.csv"
        ),
        index=False
    )

    print()
    print("=" * 75)
    print("RESULTS SAVED")
    print("=" * 75)

    print(
        "model/results/"
        "sensor_specific_results.csv"
    )

    print(
        "model/results/"
        "sensor_combination_results.csv"
    )

    print()
    print("=" * 75)
    print("SENSOR SCORE SEARCH COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()