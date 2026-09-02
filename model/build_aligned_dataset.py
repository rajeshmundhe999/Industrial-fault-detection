import os
import numpy as np


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

FAULT_FILE = os.path.join(
    DATA_DIR,
    "batch_faults.npy"
)

SCALER_MIN_FILE = os.path.join(
    DATA_DIR,
    "scaler_min.npy"
)

SCALER_SCALE_FILE = os.path.join(
    DATA_DIR,
    "scaler_scale.npy"
)

WINDOW_SIZE = 60
STRIDE = 5

TRAIN_BATCHES = list(range(1, 71))
VAL_BATCHES = list(range(71, 81))
TEST_BATCHES = list(range(81, 101))


def load_data():

    batch_data = np.load(
        BATCH_DATA_FILE,
        allow_pickle=True
    ).item()

    batch_faults = np.load(
        FAULT_FILE,
        allow_pickle=True
    ).item()

    scaler_min = np.load(
        SCALER_MIN_FILE
    )

    scaler_scale = np.load(
        SCALER_SCALE_FILE
    )

    return (
        batch_data,
        batch_faults,
        scaler_min,
        scaler_scale
    )


def scale_data(
    data,
    scaler_min,
    scaler_scale
):
    """Match sklearn MinMaxScaler.transform fitted in data/preprocess.py.

    `min_` and `scale_` are transform parameters, not data-space bounds.
    The former expression inverted their meaning and made all aligned
    experiments incomparable with the GRU-AE training data.
    """
    return data * scaler_scale + scaler_min


def create_dataset(
    batch_data,
    batch_faults,
    scaler_min,
    scaler_scale,
    batch_numbers,
    label_mode
):

    X = []
    y = []
    batch_ids = []
    end_times = []
    purity = []
    fault_start = []

    for batch_number in batch_numbers:

        key = str(batch_number)

        data = batch_data[key]

        metadata = batch_faults[key]

        fault = metadata["fault"]

        times = metadata["time"]

        data = scale_data(
            data,
            scaler_min,
            scaler_scale
        )

        n = len(data)

        if n < WINDOW_SIZE:
            continue

        for start in range(
            0,
            n - WINDOW_SIZE + 1,
            STRIDE
        ):

            end = start + WINDOW_SIZE

            window = data[
                start:end
            ]

            window_faults = fault[
                start:end
            ]

            fault_mask = (
                window_faults > 0
            )

            fault_fraction = np.mean(
                fault_mask
            )

            end_fault = (
                fault[end - 1] > 0
            )

            if label_mode == "end":

                label = int(
                    end_fault
                )

            elif label_mode == "majority":

                label = int(
                    fault_fraction >= 0.50
                )

            elif label_mode == "strict":

                label = int(
                    fault_fraction >= 0.75
                )

            elif label_mode == "any":

                label = int(
                    np.any(
                        fault_mask
                    )
                )

            else:

                raise ValueError(
                    "Unknown label mode"
                )

            if np.any(fault_mask):

                indices = np.where(
                    fault_mask
                )[0]

                first_fault_index = (
                    start + indices[0]
                )

                first_fault_time = (
                    times[first_fault_index]
                )

            else:

                first_fault_time = np.nan

            X.append(
                window.astype(
                    np.float32
                )
            )

            y.append(label)

            batch_ids.append(
                batch_number
            )

            end_times.append(
                times[end - 1]
            )

            purity.append(
                fault_fraction
            )

            fault_start.append(
                first_fault_time
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(batch_ids, dtype=np.int64),
        np.asarray(end_times, dtype=np.float32),
        np.asarray(purity, dtype=np.float32),
        np.asarray(fault_start, dtype=np.float32)
    )


def save_dataset(
    name,
    dataset
):

    X, y, batches, times, purity, fault_start = dataset

    np.save(
        os.path.join(
            DATA_DIR,
            f"X_{name}_aligned.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            DATA_DIR,
            f"y_{name}_aligned.npy"
        ),
        y
    )

    np.save(
        os.path.join(
            DATA_DIR,
            f"batch_ids_{name}_aligned.npy"
        ),
        batches
    )

    np.save(
        os.path.join(
            DATA_DIR,
            f"times_{name}_aligned.npy"
        ),
        times
    )

    np.save(
        os.path.join(
            DATA_DIR,
            f"purity_{name}_aligned.npy"
        ),
        purity
    )

    np.save(
        os.path.join(
            DATA_DIR,
            f"fault_start_{name}_aligned.npy"
        ),
        fault_start
    )


def inspect_dataset(
    name,
    dataset
):

    X, y, batches, times, purity, fault_start = dataset

    print()
    print("=" * 75)
    print(name.upper())
    print("=" * 75)

    print(
        "X shape:",
        X.shape
    )

    print(
        "Normal:",
        np.sum(y == 0)
    )

    print(
        "Fault:",
        np.sum(y == 1)
    )

    print(
        "Fault percentage:",
        f"{100 * np.mean(y):.3f}%"
    )

    print(
        "Purity mean:",
        f"{np.mean(purity):.5f}"
    )

    print(
        "Purity median:",
        f"{np.median(purity):.5f}"
    )

    print(
        "Batches:",
        np.unique(batches)
    )

    print()

    if np.any(y == 1):

        print(
            "Fault windows by batch:"
        )

        for batch in np.unique(
            batches[y == 1]
        ):

            count = np.sum(
                (batches == batch)
                &
                (y == 1)
            )

            print(
                f"Batch {batch:3d}: "
                f"{count:4d}"
            )


def main():

    print("=" * 75)
    print(
        "BUILDING ALIGNED FAULT DATASETS"
    )
    print("=" * 75)

    print()

    (
        batch_data,
        batch_faults,
        scaler_min,
        scaler_scale
    ) = load_data()

    modes = [
        "end",
        "majority",
        "strict",
        "any"
    ]

    for mode in modes:

        print()
        print(
            "#" * 75
        )

        print(
            "LABEL MODE:",
            mode.upper()
        )

        print(
            "#" * 75
        )

        train = create_dataset(
            batch_data,
            batch_faults,
            scaler_min,
            scaler_scale,
            TRAIN_BATCHES,
            mode
        )

        val = create_dataset(
            batch_data,
            batch_faults,
            scaler_min,
            scaler_scale,
            VAL_BATCHES,
            mode
        )

        test = create_dataset(
            batch_data,
            batch_faults,
            scaler_min,
            scaler_scale,
            TEST_BATCHES,
            mode
        )

        inspect_dataset(
            "TRAIN",
            train
        )

        inspect_dataset(
            "VALIDATION",
            val
        )

        inspect_dataset(
            "TEST",
            test
        )

        save_dataset(
            f"train_{mode}",
            train
        )

        save_dataset(
            f"val_{mode}",
            val
        )

        save_dataset(
            f"test_{mode}",
            test
        )

    print()
    print("=" * 75)
    print(
        "ALIGNED DATASET CREATION COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
