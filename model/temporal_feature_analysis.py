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


def calculate_features(X):

    first = np.diff(
        X,
        axis=1
    )

    second = np.diff(
        first,
        axis=1
    )

    first_mean = np.mean(
        np.abs(first),
        axis=1
    )

    second_mean = np.mean(
        np.abs(second),
        axis=1
    )

    first_max = np.max(
        np.abs(first),
        axis=1
    )

    second_max = np.max(
        np.abs(second),
        axis=1
    )

    return (
        first_mean,
        second_mean,
        first_max,
        second_max
    )


def main():

    print("=" * 75)
    print("TEMPORAL FEATURE ANALYSIS")
    print("=" * 75)

    X_test = np.load(
        os.path.join(
            DATA_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            RESULT_DIR,
            "test_labels.npy"
        )
    )

    batch_ids = np.load(
        os.path.join(
            RESULT_DIR,
            "test_batch_ids.npy"
        )
    )

    sensor_names = np.load(
        os.path.join(
            DATA_DIR,
            "sensor_names.npy"
        ),
        allow_pickle=True
    )

    print(
        "\nTest:",
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

    (
        first_mean,
        second_mean,
        first_max,
        second_max
    ) = calculate_features(
        X_test
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "OVERALL TEMPORAL FEATURE SEPARATION"
    )

    print(
        "=" * 75
    )

    rows = []

    for sensor_index, sensor_name in enumerate(
        sensor_names
    ):

        normal = (
            y_test == 0
        )

        fault = (
            y_test == 1
        )

        normal_first = np.mean(
            first_mean[
                normal,
                sensor_index
            ]
        )

        fault_first = np.mean(
            first_mean[
                fault,
                sensor_index
            ]
        )

        normal_second = np.mean(
            second_mean[
                normal,
                sensor_index
            ]
        )

        fault_second = np.mean(
            second_mean[
                fault,
                sensor_index
            ]
        )

        first_ratio = (
            fault_first
            /
            (
                normal_first
                +
                1e-12
            )
        )

        second_ratio = (
            fault_second
            /
            (
                normal_second
                +
                1e-12
            )
        )

        rows.append(
            {
                "sensor": sensor_name,
                "normal_first":
                    normal_first,
                "fault_first":
                    fault_first,
                "first_ratio":
                    first_ratio,
                "normal_second":
                    normal_second,
                "fault_second":
                    fault_second,
                "second_ratio":
                    second_ratio
            }
        )

    df = pd.DataFrame(
        rows
    )

    df = df.sort_values(
        "first_ratio",
        ascending=False
    )

    print(
        df.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "HARD FAULT TEMPORAL ANALYSIS"
    )

    print(
        "=" * 75
    )

    hard_batches = [
        91,
        95,
        97
    ]

    hard_rows = []

    for batch in hard_batches:

        mask_batch = (
            batch_ids == batch
        )

        mask_fault = (
            mask_batch
            &
            (y_test == 1)
        )

        mask_normal = (
            mask_batch
            &
            (y_test == 0)
        )

        print(
            "\nBatch",
            batch
        )

        for sensor_index, sensor_name in enumerate(
            sensor_names
        ):

            if (
                not np.any(mask_fault)
                or
                not np.any(mask_normal)
            ):
                continue

            nf = np.mean(
                first_mean[
                    mask_normal,
                    sensor_index
                ]
            )

            ff = np.mean(
                first_mean[
                    mask_fault,
                    sensor_index
                ]
            )

            ns = np.mean(
                second_mean[
                    mask_normal,
                    sensor_index
                ]
            )

            fs = np.mean(
                second_mean[
                    mask_fault,
                    sensor_index
                ]
            )

            fr = (
                ff
                /
                (
                    nf
                    +
                    1e-12
                )
            )

            sr = (
                fs
                /
                (
                    ns
                    +
                    1e-12
                )
            )

            hard_rows.append(
                {
                    "batch": batch,
                    "sensor": sensor_name,
                    "first_ratio": fr,
                    "second_ratio": sr
                }
            )

        batch_df = pd.DataFrame(
            [
                x
                for x in hard_rows
                if x["batch"] == batch
            ]
        )

        batch_df = batch_df.sort_values(
            "first_ratio",
            ascending=False
        )

        print(
            batch_df.to_string(
                index=False
            )
        )

    output = pd.DataFrame(
        hard_rows
    )

    output_path = os.path.join(
        RESULT_DIR,
        "temporal_feature_analysis.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved:",
        output_path
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "TEMPORAL FEATURE ANALYSIS COMPLETE"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()