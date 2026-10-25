"""
TEP Preprocessing — Tennessee Eastman Process benchmark.

Data layout (Braatz lab format):
  - Training files d00-d21.dat: stored as (52 vars × T timesteps), must transpose.
    * d00.dat: 500 timesteps, all normal operation
    * d01-d21.dat: 480 timesteps, fault injected at timestep 20 (0-indexed)
  - Test files d00_te.dat – d21_te.dat: (T × 52), already correct orientation
    * 960 timesteps; fault injected at timestep 160 in fault files
  - Columns 0-21 : 22 XMEAS process variables
  - Faults 3 and 9 are unmeasured → skip from test evaluation

Window parameters: size=60, stride=5 (test); stride=1 (train, to maximise data)
Labels: 1 if window-end timestep >= fault_step, else 0

Scaler: MinMaxScaler fitted on the training normal portion only.

Training data augmentation:
  - d00.dat (all 500 timesteps, 80% train / 20% val)
  - Pre-fault portion (first 20 timesteps) of d01-d21.dat training files
    appended to training set.  These are normal process operation windows.

Outputs (tep/processed/):
  X_train.npy, X_val.npy, X_test.npy, y_test.npy,
  batch_ids_test.npy, scaler_min.npy, scaler_scale.npy
"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

RAW  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
os.makedirs(OUT, exist_ok=True)

WINDOW           = 60
STRIDE_TRAIN     = 1    # dense windowing for training (more samples from short series)
STRIDE_TEST      = 5    # as specified
N_SENSORS        = 22   # XMEAS 1-22 = columns 0-21
FAULT_STEP_TRAIN = 20   # fault injection timestep in d01-d21.dat
FAULT_STEP_TEST  = 160  # fault injection timestep in d??_te.dat
SKIP_FAULTS      = {3, 9}


def load_dat(path):
    """Return (T, 52) array.

    Training files (d00-d21.dat) are stored transposed: (52, T) → transpose.
    Test files (_te.dat) are stored as (T, 52) → return as-is.
    """
    raw = np.loadtxt(path)
    if raw.shape[0] == 52 and raw.ndim == 2 and raw.shape[1] != 52:
        return raw.T
    return raw


def make_windows(data, window=WINDOW, stride=STRIDE_TRAIN):
    """Slide window; return (N, window, n_sensors)."""
    wins = []
    T = len(data)
    for start in range(0, T - window + 1, stride):
        wins.append(data[start:start + window])
    return np.asarray(wins, dtype=np.float32)


def make_labels_test(T, fault_step, is_fault_file):
    """Label = 1 if window-end index >= fault_step."""
    labels = []
    for start in range(0, T - WINDOW + 1, STRIDE_TEST):
        end_idx = start + WINDOW - 1
        labels.append(int(is_fault_file and end_idx >= fault_step))
    return np.asarray(labels, dtype=np.int32)


def main():
    print("=" * 60)
    print("TEP PREPROCESSING")
    print("=" * 60)

    # ── Load normal training file (d00.dat) ───────────────────────────────────
    d00_path = os.path.join(RAW, "d00.dat")
    raw_d00  = load_dat(d00_path)[:, :N_SENSORS]   # (500, 22)
    T_all    = len(raw_d00)
    split    = int(T_all * 0.8)                     # 400 train, 100 val
    raw_tr   = raw_d00[:split]
    raw_va   = raw_d00[split:]
    print(f"d00.dat  T={T_all}  train={split}  val={T_all - split}")

    # ── MinMaxScaler fitted on d00 training portion only ─────────────────────
    scaler = MinMaxScaler()
    scaler.fit(raw_tr)
    sc_min   = scaler.data_min_.astype(np.float32)
    sc_scale = (scaler.data_max_ - scaler.data_min_).astype(np.float32)
    sc_scale[sc_scale == 0] = 1.0

    def normalize(x):
        return ((x - sc_min) / sc_scale).astype(np.float32)

    # ── Build training windows (d00 training portion) ─────────────────────────
    tr_wins = make_windows(normalize(raw_tr), stride=STRIDE_TRAIN)
    va_wins = make_windows(normalize(raw_va), stride=STRIDE_TRAIN)   # stride=1 for val too
    print(f"d00 train windows: {len(tr_wins)}  val windows: {len(va_wins)}")

    # ── Augment training with pre-fault normal data from d01-d21.dat ──────────
    # The first FAULT_STEP_TRAIN timesteps of each fault training file are
    # normal operation. We include only windows that end BEFORE the fault step.
    extra_segs = []
    for fid in range(1, 22):
        fpath = os.path.join(RAW, f"d{fid:02d}.dat")
        if not os.path.exists(fpath):
            continue
        raw_f = load_dat(fpath)[:, :N_SENSORS]   # (480, 22)
        # Keep only the pre-fault portion (rows 0..FAULT_STEP_TRAIN-1)
        pre_fault = raw_f[:FAULT_STEP_TRAIN]
        if len(pre_fault) >= WINDOW:
            extra_segs.append(normalize(pre_fault))
            seg_wins = len(make_windows(normalize(pre_fault), stride=STRIDE_TRAIN))
            print(f"  d{fid:02d}.dat  pre-fault rows={len(pre_fault)}  extra_wins={seg_wins}")
        else:
            # pre-fault shorter than window — pad or skip; just use all rows
            # as augmentation at timestep level by appending to d00 normal series
            print(f"  d{fid:02d}.dat  pre-fault too short ({len(pre_fault)}<{WINDOW}), skipping")

    # Concatenate all normal segments for training
    all_tr_segs = [normalize(raw_tr)] + extra_segs
    combined_tr  = np.vstack(all_tr_segs)   # (T_combined, 22)
    # We cannot window across segment boundaries, so window each separately
    all_tr_wins_list = [tr_wins]
    for seg in extra_segs:
        w = make_windows(seg, stride=STRIDE_TRAIN)
        if len(w) > 0:
            all_tr_wins_list.append(w)

    X_train = np.concatenate(all_tr_wins_list, axis=0)
    X_val   = va_wins
    print(f"\nFinal X_train: {X_train.shape}   X_val: {X_val.shape}")

    # ── Test data ─────────────────────────────────────────────────────────────
    # batch_id 0 → d00_te.dat (normal), batch_id i → d{fid}_te.dat (fault)
    test_fault_ids = [0] + [k for k in range(1, 22) if k not in SKIP_FAULTS]
    print(f"\nTest fault IDs: {test_fault_ids}")

    X_test_list, y_test_list, bid_list = [], [], []

    for bid, fid in enumerate(test_fault_ids):
        fname = f"d{fid:02d}_te.dat"
        fpath = os.path.join(RAW, fname)
        if not os.path.exists(fpath):
            print(f"  MISSING {fname} — skipping")
            continue
        raw   = load_dat(fpath)[:, :N_SENSORS]   # (960, 22)
        T     = len(raw)
        normed = normalize(raw)
        is_fault = (fid != 0)
        wins   = make_windows(normed, stride=STRIDE_TEST)
        labels = make_labels_test(T, FAULT_STEP_TEST, is_fault)
        bids   = np.full(len(wins), bid, dtype=np.int32)

        X_test_list.append(wins)
        y_test_list.append(labels)
        bid_list.append(bids)
        print(f"  {fname}  T={T}  windows={len(wins)}  fault_wins={labels.sum()}")

    X_test    = np.concatenate(X_test_list, axis=0)
    y_test    = np.concatenate(y_test_list, axis=0)
    batch_ids = np.concatenate(bid_list,    axis=0)

    print(f"\nX_test: {X_test.shape}  fault_windows={y_test.sum()}/{len(y_test)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(OUT, "X_train.npy"),        X_train)
    np.save(os.path.join(OUT, "X_val.npy"),          X_val)
    np.save(os.path.join(OUT, "X_test.npy"),         X_test)
    np.save(os.path.join(OUT, "y_test.npy"),         y_test)
    np.save(os.path.join(OUT, "batch_ids_test.npy"), batch_ids)
    np.save(os.path.join(OUT, "scaler_min.npy"),     sc_min)
    np.save(os.path.join(OUT, "scaler_scale.npy"),   sc_scale)

    print("\nSaved to:", OUT)
    print("Done.")


if __name__ == "__main__":
    main()
