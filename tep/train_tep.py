"""
Train GRU-AE v2 on TEP normal data (22 sensor channels).

Mirrors the training protocol from model/train_gru_v2.py:
  - MSE loss, AdamW, CosineAnnealingWarmRestarts
  - Gaussian noise augmentation (sigma=0.01)
  - Gradient clipping at 1.0
  - Patience=40 early stopping
  - input_size=22 (TEP has 22 measured process variables)
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── paths ────────────────────────────────────────────────────────────────────
TEP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE     = os.path.dirname(TEP_DIR)
DATA     = os.path.join(TEP_DIR, "processed")
SAVED    = os.path.join(TEP_DIR, "saved")
RESULTS  = os.path.join(TEP_DIR, "results")
sys.path.insert(0, BASE)

from model.gru_ae_v2 import GRUAutoencoderV2

os.makedirs(SAVED,   exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 400
BATCH_SIZE  = 256
LR          = 5e-4
PATIENCE    = 40
NOISE_SIGMA = 0.01
INPUT_SIZE  = 22    # TEP: 22 measured process variables
BEST_PATH   = os.path.join(SAVED, "gru_ae_v2_tep_best.pt")


def add_noise(x, sigma=NOISE_SIGMA):
    return x + sigma * torch.randn_like(x)


def main():
    print("=" * 60)
    print("GRU-AE V2 TEP TRAINING")
    print("=" * 60)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ── data ─────────────────────────────────────────────────────────────────
    X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(DATA, "X_val.npy")).astype(np.float32)
    print(f"Train: {X_train.shape}   Val: {X_val.shape}")

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          pin_memory=True, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    model = GRUAutoencoderV2(input_size=INPUT_SIZE, dropout=0.15).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    best_val   = float("inf")
    no_improve = 0
    history    = []

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tr_loss = 0.0
        for (x,) in train_dl:
            x  = x.to(DEVICE, non_blocking=True)
            xn = add_noise(x)
            recon = model(xn)
            loss  = criterion(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_dl.dataset)

        # validate
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for (x,) in val_dl:
                x = x.to(DEVICE, non_blocking=True)
                va_loss += criterion(model(x), x).item() * x.size(0)
        va_loss /= len(val_dl.dataset)

        scheduler.step()
        history.append([epoch, tr_loss, va_loss])

        if epoch % 20 == 0 or epoch <= 5:
            print(f"Epoch {epoch:03d}/{EPOCHS}  "
                  f"train={tr_loss:.6f}  val={va_loss:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss": va_loss, "epoch": epoch,
                        "input_size": INPUT_SIZE}, BEST_PATH)
            if epoch % 20 == 0 or epoch <= 5:
                print("  >> new best saved")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

    np.save(os.path.join(RESULTS, "tep_training_history.npy"),
            np.asarray(history))
    print(f"\nBest val loss: {best_val:.6f}")
    print("Saved:", BEST_PATH)


if __name__ == "__main__":
    main()
