import os
import sys
import numpy as np
import torch
import pandas as pd

from sklearn.metrics import (
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

THRESHOLD = 0.0003591771


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

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

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

    all_errors = []

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

            errors = (
                batch - reconstructed
            ) ** 2

            all_errors.append(
                errors.cpu().numpy()
            )

    return np.concatenate(
        all_errors,
        axis=0
    )


def main():

    print("=" * 75)
    print("GRU-AE MISSED FAULT ANALYSIS")
    print("=" * 75)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
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

    print()
    print(
        "Test windows:",
        X_test.shape
    )

    print(
        "Fault windows:",
        int(np.sum(y_test == 1))
    )

    print()
    print("Loading model...")

    model = load_model(
        device
    )

    print("Model loaded.")

    print()
    print("Calculating reconstruction errors...")

    errors = calculate_errors(
        model,
        X_test,
        device
    )

    scores = errors.mean(
        axis=(1, 2)
    )

    predictions = (
        scores >= THRESHOLD
    ).astype(int)

    print("Errors calculated.")

    print()
    print("=" * 75)
    print("OVERALL PERFORMANCE")
    print("=" * 75)

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

    print(
        f"Threshold : {THRESHOLD:.10f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print()
    print("=" * 75)
    print("BATCH-WISE DETECTION")
    print("=" * 75)

    fault_batches = sorted(
        np.unique(
            batch_ids[y_test == 1]
        )
    )

    batch_results = []

    print()
    print(
        "Batch | Fault | Detected | Missed | Recall | "
        "Fault Mean | Missed Mean"
    )

    print("-" * 90)

    for batch in fault_batches:

        mask = (
            batch_ids == batch
        )

        fault_mask = (
            mask &
            (y_test == 1)
        )

        detected_mask = (
            fault_mask &
            (predictions == 1)
        )

        missed_mask = (
            fault_mask &
            (predictions == 0)
        )

        fault_count = int(
            np.sum(fault_mask)
        )

        detected_count = int(
            np.sum(detected_mask)
        )

        missed_count = int(
            np.sum(missed_mask)
        )

        batch_recall = (
            detected_count / fault_count
            if fault_count > 0
            else 0
        )

        fault_mean = (
            float(
                np.mean(
                    scores[fault_mask]
                )
            )
            if fault_count > 0
            else 0
        )

        missed_mean = (
            float(
                np.mean(
                    scores[missed_mask]
                )
            )
            if missed_count > 0
            else 0
        )

        print(
            f"{int(batch):5d} | "
            f"{fault_count:5d} | "
            f"{detected_count:8d} | "
            f"{missed_count:6d} | "
            f"{batch_recall:.4f} | "
            f"{fault_mean:.8f} | "
            f"{missed_mean:.8f}"
        )

        batch_results.append(
            {
                "batch": int(batch),
                "fault_windows": fault_count,
                "detected": detected_count,
                "missed": missed_count,
                "recall": batch_recall,
                "fault_mean_score": fault_mean,
                "missed_mean_score": missed_mean
            }
        )

    print()
    print("=" * 75)
    print("MISSED FAULT WINDOWS")
    print("=" * 75)

    missed_indices = np.where(
        (y_test == 1) &
        (predictions == 0)
    )[0]

    print()
    print(
        "Total missed:",
        len(missed_indices)
    )

    print()
    print(
        "Index | Batch | Time | Score | "
        "Fault Purity | Highest Sensor Error"
    )

    print("-" * 100)

    fault_data = np.load(
        os.path.join(
            DATA_DIR,
            "batch_faults.npy"
        ),
        allow_pickle=True
    ).item()

    missed_results = []

    for index in missed_indices:

        batch = int(
            batch_ids[index]
        )

        time = float(
            times[index]
        )

        score = float(
            scores[index]
        )

        metadata = fault_data[
            str(batch)
        ]

        raw_times = metadata[
            "time"
        ]

        raw_faults = metadata[
            "fault"
        ]

        nearest = np.argmin(
            np.abs(
                raw_times - time
            )
        )

        fault_start = max(
            0,
            nearest - 59
        )

        fault_end = min(
            len(raw_faults),
            nearest + 1
        )

        window_faults = raw_faults[
            fault_start:fault_end
        ]

        purity = float(
            np.mean(
                window_faults > 0
            )
        )

        sensor_errors = errors[
            index
        ].mean(
            axis=0
        )

        highest_sensor = int(
            np.argmax(
                sensor_errors
            )
        )

        print(
            f"{index:5d} | "
            f"{batch:5d} | "
            f"{time:5.1f} | "
            f"{score:.8f} | "
            f"{purity:.3f} | "
            f"{SENSORS[highest_sensor]}"
        )

        missed_results.append(
            {
                "index": int(index),
                "batch": batch,
                "time": time,
                "score": score,
                "fault_purity": purity,
                "highest_sensor": SENSORS[
                    highest_sensor
                ]
            }
        )

    print()
    print("=" * 75)
    print("MISSED FAULTS BY PURITY")
    print("=" * 75)

    missed_purities = np.array(
        [
            x["fault_purity"]
            for x in missed_results
        ]
    )

    if len(missed_purities) > 0:

        print(
            "Mean purity:",
            f"{missed_purities.mean():.4f}"
        )

        print(
            "Median purity:",
            f"{np.median(missed_purities):.4f}"
        )

        print(
            ">= 10%:",
            int(
                np.sum(
                    missed_purities >= 0.10
                )
            )
        )

        print(
            ">= 25%:",
            int(
                np.sum(
                    missed_purities >= 0.25
                )
            )
        )

        print(
            ">= 50%:",
            int(
                np.sum(
                    missed_purities >= 0.50
                )
            )
        )

        print(
            ">= 75%:",
            int(
                np.sum(
                    missed_purities >= 0.75
                )
            )
        )

    print()
    print("=" * 75)
    print("MISSED FAULTS BY SENSOR")
    print("=" * 75)

    sensor_counts = {}

    for item in missed_results:

        sensor = item[
            "highest_sensor"
        ]

        if sensor not in sensor_counts:
            sensor_counts[sensor] = 0

        sensor_counts[sensor] += 1

    sorted_sensors = sorted(
        sensor_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for sensor, count in sorted_sensors:

        print(
            f"{sensor:35s} : {count}"
        )

    print()
    print("=" * 75)
    print("LOW-SCORE FAULT WINDOWS")
    print("=" * 75)

    fault_indices = np.where(
        y_test == 1
    )[0]

    sorted_fault_indices = sorted(
        fault_indices,
        key=lambda i: scores[i]
    )

    print()
    print(
        "The 30 faulty windows with the smallest"
    )

    print(
        "reconstruction errors:"
    )

    print()

    print(
        "Index | Batch | Time | Score | Predicted"
    )

    print("-" * 60)

    for index in sorted_fault_indices[:30]:

        print(
            f"{index:5d} | "
            f"{int(batch_ids[index]):5d} | "
            f"{times[index]:5.1f} | "
            f"{scores[index]:.8f} | "
            f"{predictions[index]}"
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    batch_df = pd.DataFrame(
        batch_results
    )

    batch_file = os.path.join(
        RESULTS_DIR,
        "gru_missed_fault_batch_analysis.csv"
    )

    batch_df.to_csv(
        batch_file,
        index=False
    )

    missed_df = pd.DataFrame(
        missed_results
    )

    missed_file = os.path.join(
        RESULTS_DIR,
        "gru_missed_fault_windows.csv"
    )

    missed_df.to_csv(
        missed_file,
        index=False
    )

    print()
    print("=" * 75)
    print("RESULTS SAVED")
    print("=" * 75)

    print(
        batch_file
    )

    print(
        missed_file
    )

    print()
    print("=" * 75)
    print("MISSED FAULT ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()