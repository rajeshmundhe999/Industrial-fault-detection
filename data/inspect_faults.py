import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

X_TEST = np.load(
    os.path.join(PROCESSED_DIR, "X_test.npy")
)

Y_TEST = np.load(
    os.path.join(PROCESSED_DIR, "y_test.npy")
)

BATCH_IDS = np.load(
    os.path.join(PROCESSED_DIR, "batch_ids_test.npy")
)

TIMES = np.load(
    os.path.join(PROCESSED_DIR, "times_test.npy")
)

print("=" * 60)
print("FAULT TEST SET INSPECTION")
print("=" * 60)

print()

print("Test shape:", X_TEST.shape)

print("Total test windows:", len(X_TEST))

print("Normal windows:", np.sum(Y_TEST == 0))

print("Fault windows:", np.sum(Y_TEST == 1))

print()

fault_mask = Y_TEST == 1

fault_batches = BATCH_IDS[fault_mask]

unique_batches, counts = np.unique(
    fault_batches,
    return_counts=True
)

print("Fault windows by batch:")
print()

for batch, count in zip(
    unique_batches,
    counts
):
    print(
        f"Batch {int(batch):3d}: "
        f"{int(count):4d} fault windows"
    )

print()

print("Fault batches:", len(unique_batches))

print()

fault_times = TIMES[fault_mask]

print(
    "Earliest fault window time:",
    np.min(fault_times)
)

print(
    "Latest fault window time:",
    np.max(fault_times)
)

print()

print("Fault batch IDs:")

print(
    sorted(
        [int(x) for x in unique_batches]
    )
)

print()

print("=" * 60)