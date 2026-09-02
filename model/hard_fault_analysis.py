import os
import numpy as np
import pandas as pd


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


def main():

    print("=" * 75)
    print("HARD FAULT SENSOR ANALYSIS")
    print("=" * 75)

    batches = np.load(
        os.path.join(
            DATA_DIR,
            "batch_data.npy"
        ),
        allow_pickle=True
    ).item()

    faults = np.load(
        os.path.join(
            DATA_DIR,
            "batch_faults.npy"
        ),
        allow_pickle=True
    ).item()

    sensor_names = np.load(
        os.path.join(
            DATA_DIR,
            "sensor_names.npy"
        ),
        allow_pickle=True
    )

    hard_batches = [
        "91",
        "95",
        "97"
    ]

    print(
        "\nAnalysing batches:",
        hard_batches
    )

    print(
        "\nSensors:"
    )

    for i, name in enumerate(sensor_names):

        print(
            i + 1,
            name
        )

    all_rows = []

    for batch_id in hard_batches:

        X = batches[batch_id]

        fault = faults[batch_id]["fault"]

        fault_indices = np.where(
            fault == 1
        )[0]

        normal_indices = np.where(
            fault == 0
        )[0]

        if len(fault_indices) == 0:
            continue

        fault_start = fault_indices[0]

        print(
            "\n" + "=" * 75
        )

        print(
            "BATCH",
            batch_id
        )

        print(
            "=" * 75
        )

        print(
            "Fault starts at:",
            fault_start
        )

        print(
            "Fault samples:",
            len(fault_indices)
        )

        for sensor_index, sensor_name in enumerate(
            sensor_names
        ):

            normal_values = X[
                normal_indices,
                sensor_index
            ]

            fault_values = X[
                fault_indices,
                sensor_index
            ]

            normal_mean = np.mean(
                normal_values
            )

            fault_mean = np.mean(
                fault_values
            )

            normal_std = np.std(
                normal_values
            )

            fault_std = np.std(
                fault_values
            )

            mean_difference = (
                fault_mean
                -
                normal_mean
            )

            normalized_difference = (
                abs(mean_difference)
                /
                (
                    normal_std
                    +
                    1e-8
                )
            )

            normal_min = np.min(
                normal_values
            )

            normal_max = np.max(
                normal_values
            )

            fault_min = np.min(
                fault_values
            )

            fault_max = np.max(
                fault_values
            )

            fault_range = (
                fault_max
                -
                fault_min
            )

            normal_range = (
                normal_max
                -
                normal_min
            )

            early_end = min(
                fault_start + 20,
                len(X)
            )

            early_values = X[
                fault_start:early_end,
                sensor_index
            ]

            if len(early_values) > 1:

                early_change = (
                    early_values[-1]
                    -
                    early_values[0]
                )

            else:

                early_change = 0.0

            row = {
                "batch": int(batch_id),
                "sensor": sensor_name,
                "normal_mean": normal_mean,
                "fault_mean": fault_mean,
                "normal_std": normal_std,
                "fault_std": fault_std,
                "mean_difference": mean_difference,
                "normalized_difference":
                    normalized_difference,
                "normal_range": normal_range,
                "fault_range": fault_range,
                "early_change": early_change
            }

            all_rows.append(
                row
            )

        batch_df = pd.DataFrame(
            [
                x
                for x in all_rows
                if x["batch"] == int(batch_id)
            ]
        )

        batch_df = batch_df.sort_values(
            "normalized_difference",
            ascending=False
        )

        print(
            "\nSensor ranking:"
        )

        print(
            batch_df[
                [
                    "sensor",
                    "normalized_difference",
                    "mean_difference",
                    "early_change"
                ]
            ].to_string(
                index=False
            )
        )

    output = pd.DataFrame(
        all_rows
    )

    output_path = os.path.join(
        RESULT_DIR,
        "hard_fault_sensor_analysis.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "OVERALL HARD-FAULT SENSOR RANKING"
    )

    print(
        "=" * 75
    )

    summary = (
        output
        .groupby("sensor")
        [
            "normalized_difference"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        summary.to_string()
    )

    print(
        "\nSaved:",
        output_path
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "HARD FAULT ANALYSIS COMPLETE"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()