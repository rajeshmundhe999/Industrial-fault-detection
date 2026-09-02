import numpy as np

faults = np.load(
    "data/processed/batch_faults.npy",
    allow_pickle=True
).item()

y = np.load("data/processed/y_test.npy")
batch_ids = np.load("data/processed/batch_ids_test.npy")
times = np.load("data/processed/times_test.npy")

print("=" * 70)
print("RAW FAULT TIMES")
print("=" * 70)

for batch in ["91", "92", "93", "94", "95", "96", "97", "98", "99", "100"]:

    fault = faults[batch]["fault"]
    time = faults[batch]["time"]

    indices = np.where(fault > 0)[0]

    if len(indices) == 0:
        continue

    print(
        f"Batch {batch}: "
        f"first={time[indices[0]]:.1f}, "
        f"last={time[indices[-1]]:.1f}, "
        f"samples={len(indices)}"
    )


print()
print("=" * 70)
print("FIRST 30 FAULT-LABELLED TEST WINDOWS")
print("=" * 70)

indices = np.where(y == 1)[0][:30]

for i in indices:

    batch = str(batch_ids[i])
    end_time = times[i]

    fault = faults[batch]["fault"]
    fault_time = faults[batch]["time"]

    nearest = np.argmin(
        np.abs(fault_time - end_time)
    )

    print(
        f"Window {i:4d} | "
        f"Batch {batch:3s} | "
        f"Stored time {end_time:7.1f} | "
        f"Nearest fault value {fault[nearest]:.1f}"
    )


print()
print("=" * 70)
print("LABEL DISTRIBUTION BY BATCH")
print("=" * 70)

for batch in sorted(np.unique(batch_ids)):

    mask = batch_ids == batch

    print(
        f"Batch {batch:3d} | "
        f"Windows {np.sum(mask):4d} | "
        f"Fault labels {np.sum(y[mask]):4d}"
    )