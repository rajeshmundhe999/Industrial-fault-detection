import numpy as np

DATA = "data/processed"

batch_faults = np.load(
    DATA + "/batch_faults.npy",
    allow_pickle=True
).item()

X_test = np.load(
    DATA + "/X_test.npy"
)

y_test = np.load(
    DATA + "/y_test.npy"
)

batch_ids = np.load(
    DATA + "/batch_ids_test.npy"
)

times = np.load(
    DATA + "/times_test.npy"
)

print("=" * 75)
print("WINDOW LABEL ALIGNMENT ANALYSIS")
print("=" * 75)

print("\nTest windows:", len(X_test))
print("Labels:", len(y_test))

print("\nLabel distribution:")
print("Normal:", np.sum(y_test == 0))
print("Fault :", np.sum(y_test == 1))

print("\n" + "=" * 75)
print("FAULT WINDOW INSPECTION")
print("=" * 75)

fault_indices = np.where(y_test == 1)[0]

for i in fault_indices[:20]:

    batch = str(batch_ids[i])

    fault_array = batch_faults[batch]["fault"]

    start = int(times[i])

    end = start + X_test.shape[1] - 1

    actual_start = max(0, start)
    actual_end = min(
        len(fault_array),
        end + 1
    )

    window_faults = fault_array[
        actual_start:actual_end
    ]

    fault_count = int(
        np.sum(window_faults)
    )

    total = len(window_faults)

    print(
        f"Window {i:4d} | "
        f"Batch {batch:>3s} | "
        f"Time {start:6.1f}-{end:6.1f} | "
        f"Fault samples {fault_count:3d}/{total:3d}"
    )

print("\n" + "=" * 75)
print("FAULT WINDOW PURITY")
print("=" * 75)

purities = []

for i in fault_indices:

    batch = str(batch_ids[i])

    fault_array = batch_faults[batch]["fault"]

    start = int(times[i])

    end = start + X_test.shape[1]

    window_faults = fault_array[
        max(0, start):min(
            len(fault_array),
            end
        )
    ]

    if len(window_faults) > 0:

        purity = np.mean(
            window_faults
        )

        purities.append(purity)

purities = np.asarray(purities)

print(
    "Minimum fault purity:",
    purities.min()
)

print(
    "Mean fault purity:",
    purities.mean()
)

print(
    "Median fault purity:",
    np.median(purities)
)

print(
    "Maximum fault purity:",
    purities.max()
)

print("\nPurity groups:")

for threshold in [
    0.01,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.99
]:

    print(
        f"< {threshold:4.2f}:",
        np.sum(purities < threshold)
    )

print("\n" + "=" * 75)
print("WINDOW LABEL ANALYSIS COMPLETE")
print("=" * 75)