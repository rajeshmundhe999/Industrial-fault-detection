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

BATCH_DATA_FILE = os.path.join(
    DATA_DIR,
    "batch_data.npy"
)

BATCH_FAULT_FILE = os.path.join(
    DATA_DIR,
    "batch_faults.npy"
)

X_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "X_train.npy"
)

X_VAL_FILE = os.path.join(
    DATA_DIR,
    "X_val.npy"
)

X_TEST_FILE = os.path.join(
    DATA_DIR,
    "X_test.npy"
)

Y_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "y_train.npy"
)

Y_VAL_FILE = os.path.join(
    DATA_DIR,
    "y_val.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "y_test.npy"
)

BATCH_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "batch_ids_train.npy"
)

BATCH_VAL_FILE = os.path.join(
    DATA_DIR,
    "batch_ids_val.npy"
)

BATCH_TEST_FILE = os.path.join(
    DATA_DIR,
    "batch_ids_test.npy"
)

TIME_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "times_train.npy"
)

TIME_VAL_FILE = os.path.join(
    DATA_DIR,
    "times_val.npy"
)

TIME_TEST_FILE = os.path.join(
    DATA_DIR,
    "times_test.npy"
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

WINDOW_SIZE = 60
STRIDE = 5


def load_data():

    print(
        "Loading processed batch data..."
    )

    batch_data = np.load(
        BATCH_DATA_FILE,
        allow_pickle=True
    ).item()

    batch_faults = np.load(
        BATCH_FAULT_FILE,
        allow_pickle=True
    ).item()

    return batch_data, batch_faults


def audit_raw_batches(
    batch_data,
    batch_faults
):

    print()
    print("=" * 90)
    print("RAW BATCH AUDIT")
    print("=" * 90)

    rows = []

    for batch_number in range(1, 101):

        key = str(batch_number)

        data = batch_data[key]

        metadata = batch_faults[key]

        times = metadata["time"]

        faults = metadata["fault"]

        fault_indices = np.where(
            faults > 0
        )[0]

        if len(fault_indices) > 0:

            first_fault = times[
                fault_indices[0]
            ]

            last_fault = times[
                fault_indices[-1]
            ]

            fault_samples = len(
                fault_indices
            )

        else:

            first_fault = np.nan
            last_fault = np.nan
            fault_samples = 0

        rows.append({
            "batch": batch_number,
            "samples": len(data),
            "start_time": np.min(times),
            "end_time": np.max(times),
            "fault_samples": fault_samples,
            "fault_fraction":
                fault_samples / len(data),
            "first_fault":
                first_fault,
            "last_fault":
                last_fault
        })

    df = pd.DataFrame(rows)

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print(
        "Faulty batches:",
        df[
            df["fault_samples"] > 0
        ]["batch"].tolist()
    )

    print(
        "Normal batches:",
        df[
            df["fault_samples"] == 0
        ]["batch"].tolist()
    )

    return df


def audit_current_split():

    print()
    print("=" * 90)
    print("CURRENT DATASET SPLIT")
    print("=" * 90)

    files = [
        (
            "TRAIN",
            X_TRAIN_FILE,
            Y_TRAIN_FILE,
            BATCH_TRAIN_FILE,
            TIME_TRAIN_FILE
        ),
        (
            "VALIDATION",
            X_VAL_FILE,
            Y_VAL_FILE,
            BATCH_VAL_FILE,
            TIME_VAL_FILE
        ),
        (
            "TEST",
            X_TEST_FILE,
            Y_TEST_FILE,
            BATCH_TEST_FILE,
            TIME_TEST_FILE
        )
    ]

    for name, xf, yf, bf, tf in files:

        X = np.load(xf)

        y = np.load(yf)

        batches = np.load(bf)

        times = np.load(tf)

        print()
        print(
            name
        )

        print(
            "Shape:",
            X.shape
        )

        print(
            "Labels:",
            np.bincount(
                y,
                minlength=2
            )
        )

        print(
            "Unique batches:",
            np.unique(
                batches
            )
        )

        print(
            "Time range:",
            np.min(times),
            "to",
            np.max(times)
        )


def calculate_window_statistics(
    batch_data,
    batch_faults,
    batches
):

    results = []

    for batch_number in batches:

        key = str(batch_number)

        data = batch_data[key]

        metadata = batch_faults[key]

        fault = metadata["fault"]

        times = metadata["time"]

        n = len(data)

        if n < WINDOW_SIZE:
            continue

        for start in range(
            0,
            n - WINDOW_SIZE + 1,
            STRIDE
        ):

            end = start + WINDOW_SIZE

            window_faults = (
                fault[start:end]
            )

            purity = np.mean(
                window_faults > 0
            )

            fault_count = np.sum(
                window_faults > 0
            )

            results.append({
                "batch": batch_number,
                "start_index": start,
                "end_index": end - 1,
                "start_time":
                    times[start],
                "end_time":
                    times[end - 1],
                "fault_count":
                    fault_count,
                "purity":
                    purity
            })

    return pd.DataFrame(
        results
    )


def audit_window_purity(
    batch_data,
    batch_faults
):

    print()
    print("=" * 90)
    print("WINDOW PURITY AUDIT")
    print("=" * 90)

    all_batches = list(
        range(1, 101)
    )

    df = calculate_window_statistics(
        batch_data,
        batch_faults,
        all_batches
    )

    print()
    print(
        "Total windows:",
        len(df)
    )

    print(
        "Window size:",
        WINDOW_SIZE
    )

    print(
        "Stride:",
        STRIDE
    )

    print()

    print(
        "Purity statistics:"
    )

    print(
        df["purity"].describe()
    )

    print()

    print(
        "Fault windows under different definitions:"
    )

    definitions = [
        (
            "ANY",
            0.0,
            True
        ),
        (
            ">=10%",
            0.10,
            False
        ),
        (
            ">=25%",
            0.25,
            False
        ),
        (
            ">=50%",
            0.50,
            False
        ),
        (
            ">=75%",
            0.75,
            False
        ),
        (
            "100%",
            1.00,
            False
        )
    ]

    for name, threshold, any_rule in definitions:

        if any_rule:

            mask = (
                df["fault_count"] > 0
            )

        else:

            mask = (
                df["purity"] >= threshold
            )

        print(
            f"{name:>8}: "
            f"{int(mask.sum()):>6} windows "
            f"({100 * mask.mean():.2f}%)"
        )

    print()
    print(
        "Fault-window purity distribution:"
    )

    fault_windows = df[
        df["fault_count"] > 0
    ]

    bins = [
        -0.001,
        0.01,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        1.001
    ]

    labels = [
        "<1%",
        "1-10%",
        "10-25%",
        "25-50%",
        "50-75%",
        "75-90%",
        "90-100%"
    ]

    groups = pd.cut(
        fault_windows["purity"],
        bins=bins,
        labels=labels
    )

    print(
        groups.value_counts(
            sort=False
        )
    )

    return df


def audit_sensor_quality(
    batch_data
):

    print()
    print("=" * 90)
    print("SENSOR QUALITY AUDIT")
    print("=" * 90)

    all_data = []

    for batch_number in range(
        1,
        101
    ):

        all_data.append(
            batch_data[
                str(batch_number)
            ]
        )

    all_data = np.concatenate(
        all_data,
        axis=0
    )

    rows = []

    for i, sensor in enumerate(
        SENSORS
    ):

        values = all_data[:, i]

        finite = np.isfinite(
            values
        )

        valid = values[
            finite
        ]

        rows.append({
            "sensor": sensor,
            "min": np.min(valid),
            "max": np.max(valid),
            "mean": np.mean(valid),
            "std": np.std(valid),
            "missing": np.sum(
                ~finite
            ),
            "unique": len(
                np.unique(valid)
            ),
            "zero_fraction":
                np.mean(valid == 0)
        })

    df = pd.DataFrame(rows)

    print(
        df.to_string(
            index=False
        )
    )

    return df


def audit_fault_distribution(
    batch_data,
    batch_faults
):

    print()
    print("=" * 90)
    print("FAULT DISTRIBUTION AUDIT")
    print("=" * 90)

    faulty_batches = []

    for batch_number in range(
        1,
        101
    ):

        fault = batch_faults[
            str(batch_number)
        ]["fault"]

        if np.any(
            fault > 0
        ):

            faulty_batches.append(
                batch_number
            )

    print()
    print(
        "Faulty batches:",
        faulty_batches
    )

    print()

    for batch_number in faulty_batches:

        metadata = batch_faults[
            str(batch_number)
        ]

        fault = metadata["fault"]

        times = metadata["time"]

        indices = np.where(
            fault > 0
        )[0]

        print(
            f"Batch {batch_number:3d} | "
            f"Fault samples: "
            f"{len(indices):4d} | "
            f"Start: "
            f"{times[indices[0]]:7.2f} | "
            f"End: "
            f"{times[indices[-1]]:7.2f} | "
            f"Duration: "
            f"{times[indices[-1]] - times[indices[0]]:7.2f}"
        )


def check_split_leakage():

    print()
    print("=" * 90)
    print("SPLIT LEAKAGE CHECK")
    print("=" * 90)

    train = np.load(
        BATCH_TRAIN_FILE
    )

    val = np.load(
        BATCH_VAL_FILE
    )

    test = np.load(
        BATCH_TEST_FILE
    )

    train_set = set(
        np.unique(train)
    )

    val_set = set(
        np.unique(val)
    )

    test_set = set(
        np.unique(test)
    )

    print(
        "Train ∩ Validation:",
        train_set & val_set
    )

    print(
        "Train ∩ Test:",
        train_set & test_set
    )

    print(
        "Validation ∩ Test:",
        val_set & test_set
    )

    print()

    print(
        "Current split is batch-independent:",
        len(
            train_set &
            val_set
        ) == 0
        and
        len(
            train_set &
            test_set
        ) == 0
        and
        len(
            val_set &
            test_set
        ) == 0
    )


def save_audit(
    batch_df,
    window_df,
    sensor_df
):

    output_dir = os.path.join(
        BASE_DIR,
        "model",
        "results"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    batch_file = os.path.join(
        output_dir,
        "dataset_batch_audit.csv"
    )

    window_file = os.path.join(
        output_dir,
        "dataset_window_audit.csv"
    )

    sensor_file = os.path.join(
        output_dir,
        "dataset_sensor_audit.csv"
    )

    batch_df.to_csv(
        batch_file,
        index=False
    )

    window_df.to_csv(
        window_file,
        index=False
    )

    sensor_df.to_csv(
        sensor_file,
        index=False
    )

    print()
    print(
        "Audit files saved:"
    )

    print(
        batch_file
    )

    print(
        window_file
    )

    print(
        sensor_file
    )


def main():

    print("=" * 90)
    print(
        "INDPENSIM COMPLETE DATASET AUDIT"
    )
    print("=" * 90)

    print()
    print(
        "Window size:",
        WINDOW_SIZE
    )

    print(
        "Stride:",
        STRIDE
    )

    batch_data, batch_faults = (
        load_data()
    )

    batch_df = audit_raw_batches(
        batch_data,
        batch_faults
    )

    audit_fault_distribution(
        batch_data,
        batch_faults
    )

    window_df = audit_window_purity(
        batch_data,
        batch_faults
    )

    sensor_df = audit_sensor_quality(
        batch_data
    )

    audit_current_split()

    check_split_leakage()

    save_audit(
        batch_df,
        window_df,
        sensor_df
    )

    print()
    print("=" * 90)
    print(
        "AUDIT COMPLETE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()