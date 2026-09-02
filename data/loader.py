import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "100_Batches_IndPenSim_V3.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

SENSORS = [
    "Agitator RPM(RPM:RPM)",
    "Aeration rate(Fg:L/h)",
    "Sugar feed rate(Fs:L/h)",
    "Acid flow rate(Fa:L/h)",
    "Base flow rate(Fb:L/h)",
    "Heating/cooling water flow rate(Fc:L/h)",
    "Dissolved oxygen concentration(DO2:mg/L)",
    "pH(pH:pH)",
    "Temperature(T:K)"
]

BATCH_COLUMN = "2-PAT control(PAT_ref:PAT ref)"
TIME_COLUMN = "Time (h)"
FAULT_COLUMN = "Fault reference(Fault_ref:Fault ref)"

USE_COLUMNS = [
    BATCH_COLUMN,
    TIME_COLUMN,
    FAULT_COLUMN
] + SENSORS


def load_dataset():

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_FILE}"
        )

    print("Loading IndPenSim dataset...")
    print("This may take a few minutes.")
    print()

    chunks = []

    reader = pd.read_csv(
        RAW_FILE,
        usecols=USE_COLUMNS,
        chunksize=10000,
        low_memory=True
    )

    for i, chunk in enumerate(reader):

        chunks.append(chunk)

        if (i + 1) % 10 == 0:
            print(
                f"Processed approximately "
                f"{(i + 1) * 10000:,} rows..."
            )

    data = pd.concat(
        chunks,
        ignore_index=True
    )

    return data


def organize_batches(data):

    print()
    print("Organizing batches...")

    batches = {}

    for batch_number in range(1, 101):

        batch = data[
            data[BATCH_COLUMN] == batch_number
        ].copy()

        batch = batch.sort_values(
            TIME_COLUMN
        )

        batches[batch_number] = batch

    return batches


def save_batches(batches):

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    batch_data = {}
    batch_faults = {}

    for batch_number, batch in batches.items():

        values = batch[SENSORS].to_numpy(
            dtype=np.float32
        )

        times = batch[TIME_COLUMN].to_numpy(
            dtype=np.float32
        )

        faults = batch[FAULT_COLUMN].to_numpy(
            dtype=np.float32
        )

        batch_data[str(batch_number)] = values

        batch_faults[str(batch_number)] = {
            "time": times,
            "fault": faults
        }

    np.save(
        os.path.join(
            PROCESSED_DIR,
            "batch_data.npy"
        ),
        batch_data,
        allow_pickle=True
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            "batch_faults.npy"
        ),
        batch_faults,
        allow_pickle=True
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            "sensor_names.npy"
        ),
        np.array(SENSORS),
        allow_pickle=True
    )


def print_summary(batches):

    print()
    print("=" * 60)
    print("INDPENSIM DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Total batches: {len(batches)}"
    )

    print(
        f"Number of sensors: {len(SENSORS)}"
    )

    print()

    print("Batch sizes:")

    for batch_number in range(1, 101):

        rows = len(
            batches[batch_number]
        )

        print(
            f"Batch {batch_number:3d}: "
            f"{rows:5d} rows"
        )

    print()
    print("Sensors:")

    for sensor in SENSORS:
        print(f"- {sensor}")

    print()
    print("Saved files:")
    print(
        "data/processed/batch_data.npy"
    )
    print(
        "data/processed/batch_faults.npy"
    )
    print(
        "data/processed/sensor_names.npy"
    )

    print()
    print("Loader completed successfully.")


def main():

    data = load_dataset()

    print()
    print(
        f"Total rows loaded: {len(data):,}"
    )

    batches = organize_batches(data)

    save_batches(batches)

    print_summary(batches)


if __name__ == "__main__":
    main()