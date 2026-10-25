"""
Improved TEP Training — with heavy data augmentation.

Why the original TEP AUC=0.8721 was limited:
  - d00.dat (only normal file) → 341 training windows (tiny)
  - No augmentation → model sees very limited variation of normal behaviour

Fix:
  - Keep the same 341 real windows
  - Add 5 augmented copies per window:
      1. Gaussian noise (σ=0.02)
      2. Scale jitter (per-sensor scale ±5%)
      3. Time shift (random circular roll ±10 steps)
      4. Noise + scale combined
      5. Amplitude dropout (zero out a random sensor for whole window)
  - Total effective training: ~341 × 6 = ~2000 windows
  - Longer training (500 epochs, patience=60) since data is still small

Same GRU-AE v2 architecture (22 sensors), same evaluation script.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TEP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE    = os.path.dirname(TEP_DIR)
DATA    = os.path.join(TEP_DIR, "processed")
SAVED   = os.path.join(TEP_DIR, "saved")
RESULTS = os.path.join(TEP_DIR, "results")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

os.makedirs(SAVED,   exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

EPOCHS      = 500
BATCH_SIZE  = 128    # smaller batch (less data)
LR          = 3e-4   # slightly lower LR for more stable convergence
PATIENCE    = 60
INPUT_SIZE  = 22
BEST_PATH   = os.path.join(SAVED, "gru_ae_v2_tep_aug_best.pt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment_dataset(X, factor=5):
    """
    Create `factor` augmented copies of X (N, T, S) and concat with original.
    Returns (N*(factor+1), T, S).
    """
    N, T, S = X.shape
    copies  = [X]

    # 1. Gaussian noise
    copies.append(X + np.random.normal(0, 0.02, X.shape).astype(np.float32))

    # 2. Per-sensor scale jitter (±5%)
    scales = np.random.uniform(0.95, 1.05, (N, 1, S)).astype(np.float32)
    copies.append(X * scales)

    # 3. Random circular time shift (±10 steps)
    shifts  = np.random.randint(-10, 10, N)
    shifted = np.stack([np.roll(X[i], shifts[i], axis=0) for i in range(N)])
    copies.append(shifted.astype(np.float32))

    # 4. Noise + scale combined
    copies.append((X + np.random.normal(0, 0.015, X.shape).astype(np.float32)) * scales)

    # 5. Sensor dropout (zero out one random sensor per window)
    dropped = X.copy()
    for i in range(N):
        s_idx = np.random.randint(0, S)
        dropped[i, :, s_idx] = 0.0
    copies.append(dropped)

    return np.concatenate(copies, axis=0)


def main():
    print("=" * 60)
    print("GRU-AE V2 TEP TRAINING — WITH DATA AUGMENTATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(DATA, "X_val.npy")).astype(np.float32)
    print(f"Original train: {X_train.shape}")

    # Augment training data
    X_aug = augment_dataset(X_train, factor=5)
    print(f"Augmented train: {X_aug.shape}  ({X_aug.shape[0]//X_train.shape[0]}x original)")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_aug)),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(X_val)),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = GRUAutoencoderV2(input_size=INPUT_SIZE, dropout=0.2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  dropout=0.20")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val   = float("inf")
    no_improve = 0
    history    = []
    t0         = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for (x,) in train_dl:
            x     = x.to(DEVICE)
            recon = model(x)
            loss  = criterion(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_dl.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for (x,) in val_dl:
                x = x.to(DEVICE)
                va_loss += criterion(model(x), x).item() * x.size(0)
        va_loss /= len(val_dl.dataset)
        scheduler.step()
        history.append([epoch, tr_loss, va_loss])

        if epoch % 20 == 0 or epoch <= 5:
            print(f"Ep {epoch:04d}/{EPOCHS}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"[{(time.time()-t0)/60:.1f}min]")

        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss": va_loss, "epoch": epoch,
                        "input_size": INPUT_SIZE}, BEST_PATH)
            if epoch % 20 == 0 or epoch <= 5:
                print("  >> best saved")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    np.save(os.path.join(RESULTS, "tep_aug_training_history.npy"), np.asarray(history))
    print(f"\nBest val loss : {best_val:.6f}")
    print(f"Saved         : {BEST_PATH}")
    print(f"Total time    : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
