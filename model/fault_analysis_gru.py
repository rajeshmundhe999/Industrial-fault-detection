import os
import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
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

RESULT_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "saved"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "gru_ae_best.pt"
)

X_TEST_PATH = os.path.join(
    DATA_DIR,
    "X_test.npy"
)

LABEL_PATH = os.path.join(
    RESULT_DIR,
    "test_labels.npy"
)

BATCH_PATH = os.path.join(
    RESULT_DIR,
    "test_batch_ids.npy"
)

TIME_PATH = os.path.join(
    RESULT_DIR,
    "test_times.npy"
)

SCORE_PATH = os.path.join(
    RESULT_DIR,
    "gru_test_scores.npy"
)

VAL_SCORE_PATH = os.path.join(
    RESULT_DIR,
    "gru_val_scores.npy"
)

SENSOR_NAMES = [
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

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


def calculate_errors(
    model,
    X,
    device
):

    errors = []

    batch_size = 256

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

            batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            reconstructed = model(
                batch
            )

            error = (
                batch - reconstructed
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


def main():

    print("=" * 70)
    print("GRU-AE FAULT-WISE ANALYSIS")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    X_test = np.load(
        X_TEST_PATH
    )

    labels = np.load(
        LABEL_PATH
    )

    batch_ids = np.load(
        BATCH_PATH
    )

    times = np.load(
        TIME_PATH
    )

    val_scores = np.load(
        VAL_SCORE_PATH
    )

    threshold = np.percentile(
        val_scores,
        99.9
    )

    print(
        "\nTest windows:",
        len(X_test)
    )

    print(
        "Threshold:",
        threshold
    )

    print(
        "Fault windows:",
        int(np.sum(labels == 1))
    )

    model = load_model(
        device
    )

    print(
        "\nModel loaded."
    )

    print(
        "\nCalculating reconstruction errors..."
    )

    sensor_errors = calculate_errors(
        model,
        X_test,
        device
    )

    scores = sensor_errors.mean(
        axis=1
    )

    predictions = (
        scores > threshold
    ).astype(int)

    print(
        "Errors calculated."
    )

    print("\n" + "=" * 70)
    print("OVERALL TEST PERFORMANCE")
    print("=" * 70)

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

    print(
        "\nPrecision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1:",
        round(f1, 4)
    )

    print("\n" + "=" * 70)
    print("FAULT-WISE PERFORMANCE")
    print("=" * 70)

    print(
        "\n{:>6} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
            "Batch",
            "Fault",
            "Detected",
            "Precision",
            "Recall",
            "F1"
        )
    )

    print("-" * 70)

    batch_results = []

    unique_batches = np.unique(
        batch_ids
    )

    for batch in unique_batches:

        mask = (
            batch_ids == batch
        )

        batch_labels = labels[
            mask
        ]

        batch_predictions = predictions[
            mask
        ]

        fault_count = int(
            np.sum(batch_labels == 1)
        )

        if fault_count == 0:
            continue

        detected = int(
            np.sum(
                (batch_labels == 1)
                &
                (batch_predictions == 1)
            )
        )

        p = precision_score(
            batch_labels,
            batch_predictions,
            zero_division=0
        )

        r = recall_score(
            batch_labels,
            batch_predictions,
            zero_division=0
        )

        f = f1_score(
            batch_labels,
            batch_predictions,
            zero_division=0
        )

        print(
            "{:>6} {:>10} {:>10} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                int(batch),
                fault_count,
                detected,
                p,
                r,
                f
            )
        )

        batch_results.append(
            [
                int(batch),
                fault_count,
                detected,
                p,
                r,
                f
            ]
        )

    print("\n" + "=" * 70)
    print("FAULT SCORE SEPARATION")
    print("=" * 70)

    print(
        "\n{:>6} {:>15} {:>15} {:>15} {:>15}".format(
            "Batch",
            "Normal Mean",
            "Fault Mean",
            "Fault Max",
            "Fault Detection"
        )
    )

    print("-" * 80)

    separation_results = []

    for batch in unique_batches:

        mask = (
            batch_ids == batch
        )

        fault_mask = (
            mask
            &
            (labels == 1)
        )

        normal_mask = (
            mask
            &
            (labels == 0)
        )

        if np.sum(fault_mask) == 0:
            continue

        normal_mean = (
            np.mean(
                scores[normal_mask]
            )
            if np.sum(normal_mask) > 0
            else np.nan
        )

        fault_mean = np.mean(
            scores[fault_mask]
        )

        fault_max = np.max(
            scores[fault_mask]
        )

        detection_rate = np.mean(
            predictions[fault_mask]
        )

        print(
            "{:>6} {:>15.8f} {:>15.8f} {:>15.8f} {:>14.2%}".format(
                int(batch),
                normal_mean,
                fault_mean,
                fault_max,
                detection_rate
            )
        )

        separation_results.append(
            [
                int(batch),
                normal_mean,
                fault_mean,
                fault_max,
                detection_rate
            ]
        )

    print("\n" + "=" * 70)
    print("DETECTION DELAY ANALYSIS")
    print("=" * 70)

    print(
        "\n{:>6} {:>15} {:>15} {:>15}".format(
            "Batch",
            "Fault Start",
            "First Detection",
            "Delay"
        )
    )

    print("-" * 70)

    delay_results = []

    for batch in unique_batches:

        mask = (
            batch_ids == batch
        )

        fault_indices = np.where(
            mask & (labels == 1)
        )[0]

        if len(fault_indices) == 0:
            continue

        first_fault_index = (
            fault_indices[0]
        )

        detection_indices = np.where(
            mask
            &
            (predictions == 1)
        )[0]

        detection_indices = (
            detection_indices[
                detection_indices
                >= first_fault_index
            ]
        )

        fault_start_time = times[
            first_fault_index
        ]

        if len(detection_indices) == 0:

            print(
                "{:>6} {:>15.2f} {:>15} {:>15}".format(
                    int(batch),
                    fault_start_time,
                    "NONE",
                    "N/A"
                )
            )

            delay_results.append(
                [
                    int(batch),
                    fault_start_time,
                    np.nan
                ]
            )

        else:

            first_detection = (
                detection_indices[0]
            )

            detection_time = times[
                first_detection
            ]

            delay = (
                detection_time
                -
                fault_start_time
            )

            print(
                "{:>6} {:>15.2f} {:>15.2f} {:>15.2f}".format(
                    int(batch),
                    fault_start_time,
                    detection_time,
                    delay
                )
            )

            delay_results.append(
                [
                    int(batch),
                    fault_start_time,
                    delay
                ]
            )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            "gru_fault_wise_performance.csv"
        ),
        np.array(
            batch_results
        ),
        delimiter=",",
        header=(
            "batch,"
            "fault_windows,"
            "detected_windows,"
            "precision,"
            "recall,"
            "f1"
        ),
        comments=""
    )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            "gru_fault_score_separation.csv"
        ),
        np.array(
            separation_results
        ),
        delimiter=",",
        header=(
            "batch,"
            "normal_mean,"
            "fault_mean,"
            "fault_max,"
            "detection_rate"
        ),
        comments=""
    )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            "gru_detection_delay.csv"
        ),
        np.array(
            delay_results
        ),
        delimiter=",",
        header=(
            "batch,"
            "fault_start_time,"
            "detection_delay"
        ),
        comments=""
    )

    print("\nResults saved to:")
    print(
        os.path.join(
            RESULT_DIR,
            "gru_fault_wise_performance.csv"
        )
    )

    print(
        os.path.join(
            RESULT_DIR,
            "gru_fault_score_separation.csv"
        )
    )

    print(
        os.path.join(
            RESULT_DIR,
            "gru_detection_delay.csv"
        )
    )

    print("\n" + "=" * 70)
    print("FAULT-WISE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()