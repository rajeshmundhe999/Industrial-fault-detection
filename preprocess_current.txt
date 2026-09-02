import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

BATCH_FILE = os.path.join(
    PROCESSED_DIR,
    "batch_data.npy"
)

FAULT_FILE = os.path.join(
    PROCESSED_DIR,
    "batch_faults.npy"
)

WINDOW_SIZE = 60
STRIDE = 5

TRAIN_BATCHES = list(range(1, 71))
VAL_BATCHES = list(range(71, 81))
TEST_BATCHES = list(range(81, 101))


def load_batches():

    print("Loading processed batches...")

    batch_data = np.load(
        BATCH_FILE,
        allow_pickle=True
    ).item()

    batch_faults = np.load(
        FAULT_FILE,
        allow_pickle=True
    ).item()

    return batch_data, batch_faults


def fit_scaler(batch_data):

    print()
    print("Fitting scaler using training batches only...")

    training_data = []

    for batch_number in TRAIN_BATCHES:

        data = batch_data[str(batch_number)]

        training_data.append(data)

    training_data = np.concatenate(
        training_data,
        axis=0
    )

    scaler = MinMaxScaler()

    scaler.fit(training_data)

    print("Scaler fitted.")

    return scaler


def create_windows(
    batch_data,
    batch_faults,
    batch_numbers,
    scaler
):

    X = []
    y = []
    batch_ids = []
    times = []

    for batch_number in batch_numbers:

        data = batch_data[str(batch_number)]

        metadata = batch_faults[str(batch_number)]

        fault = metadata["fault"]
        time = metadata["time"]

        data = scaler.transform(data)

        n_rows = len(data)

        if n_rows < WINDOW_SIZE:
            continue

        for start in range(
            0,
            n_rows - WINDOW_SIZE + 1,
            STRIDE
        ):

            end = start + WINDOW_SIZE

            window = data[start:end]

            window_faults = fault[start:end]

            window_time = time[end - 1]

            if np.any(window_faults > 0):
                label = 1
            else:
                label = 0

            X.append(window)
            y.append(label)
            batch_ids.append(batch_number)
            times.append(window_time)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    batch_ids = np.asarray(
        batch_ids,
        dtype=np.int64
    )

    times = np.asarray(
        times,
        dtype=np.float32
    )

    return X, y, batch_ids, times


def save_dataset(
    name,
    X,
    y,
    batch_ids,
    times
):

    np.save(
        os.path.join(
            PROCESSED_DIR,
            f"X_{name}.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            f"y_{name}.npy"
        ),
        y
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            f"batch_ids_{name}.npy"
        ),
        batch_ids
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            f"times_{name}.npy"
        ),
        times
    )


def save_scaler(scaler):

    np.save(
        os.path.join(
            PROCESSED_DIR,
            "scaler_min.npy"
        ),
        scaler.min_
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            "scaler_scale.npy"
        ),
        scaler.scale_
    )


def print_summary(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    print()
    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print()
    print("Training:")
    print("Shape:", X_train.shape)
    print("Normal windows:", np.sum(y_train == 0))
    print("Fault windows:", np.sum(y_train == 1))

    print()
    print("Validation:")
    print("Shape:", X_val.shape)
    print("Normal windows:", np.sum(y_val == 0))
    print("Fault windows:", np.sum(y_val == 1))

    print()
    print("Test:")
    print("Shape:", X_test.shape)
    print("Normal windows:", np.sum(y_test == 0))
    print("Fault windows:", np.sum(y_test == 1))

    print()
    print("Expected window shape:")
    print("(samples, 60, 9)")


def main():

    batch_data, batch_faults = load_batches()

    scaler = fit_scaler(
        batch_data
    )

    print()
    print("Creating training windows...")

    X_train, y_train, train_ids, train_times = create_windows(
        batch_data,
        batch_faults,
        TRAIN_BATCHES,
        scaler
    )

    print("Training windows:", len(X_train))

    print()
    print("Creating validation windows...")

    X_val, y_val, val_ids, val_times = create_windows(
        batch_data,
        batch_faults,
        VAL_BATCHES,
        scaler
    )

    print("Validation windows:", len(X_val))

    print()
    print("Creating test windows...")

    X_test, y_test, test_ids, test_times = create_windows(
        batch_data,
        batch_faults,
        TEST_BATCHES,
        scaler
    )

    print("Test windows:", len(X_test))

    save_dataset(
        "train",
        X_train,
        y_train,
        train_ids,
        train_times
    )

    save_dataset(
        "val",
        X_val,
        y_val,
        val_ids,
        val_times
    )

    save_dataset(
        "test",
        X_test,
        y_test,
        test_ids,
        test_times
    )

    save_scaler(scaler)

    print_summary(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    print()
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()