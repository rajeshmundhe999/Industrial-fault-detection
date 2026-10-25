"""
Train GRU-AE v2 on normal batches 1-70 (same split as v1).

Key training improvements:
  - MSE loss instead of SmoothL1 (better separation of anomaly scores)
  - Cosine annealing LR schedule with warm restarts
  - Longer patience (50 epochs) to allow the attention layers to converge
  - Data augmentation: Gaussian noise injection during training (sigma=0.01)
    to improve robustness to sensor noise without touching test data
  - Gradient clipping at 1.0
"""
import os, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")
SAVED = os.path.join(BASE, "model", "saved")
RESULTS = os.path.join(BASE, "model", "results")
sys.path.insert(0, BASE)
from model.gru_ae_v2 import GRUAutoencoderV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 400
BATCH_SIZE = 256
LR = 5e-4
PATIENCE = 50
NOISE_SIGMA = 0.01   # Gaussian noise std on normalised data
os.makedirs(SAVED, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def add_noise(x, sigma=NOISE_SIGMA):
    return x + sigma * torch.randn_like(x)


def main():
    print("=" * 60)
    print("GRU-AE V2 TRAINING")
    print("=" * 60)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ── load data ────────────────────────────────────────────────────────────
    X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(DATA, "X_val.npy")).astype(np.float32)
    print("Train:", X_train.shape, "  Val:", X_val.shape)

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    # ── model ────────────────────────────────────────────────────────────────
    model = GRUAutoencoderV2(input_size=9, dropout=0.15).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    best_val   = float("inf")
    no_improve = 0
    best_path  = os.path.join(SAVED, "gru_ae_v2_best.pt")
    history    = []

    for epoch in range(1, EPOCHS + 1):
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for (x,) in train_dl:
            x = x.to(DEVICE, non_blocking=True)
            xn = add_noise(x)           # noisy input, clean target
            recon = model(xn)
            loss = criterion(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_dl.dataset)

        # ── validate ─────────────────────────────────────────────────────────
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for (x,) in val_dl:
                x = x.to(DEVICE, non_blocking=True)
                recon = model(x)        # clean input at eval time
                va_loss += criterion(recon, x).item() * x.size(0)
        va_loss /= len(val_dl.dataset)

        scheduler.step()
        history.append([epoch, tr_loss, va_loss])

        if epoch % 20 == 0 or epoch <= 5:
            print(f"Epoch {epoch:03d}/{EPOCHS}  train={tr_loss:.6f}  val={va_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss": va_loss, "epoch": epoch, "input_size": 9}, best_path)
            if epoch % 20 == 0 or epoch <= 5:
                print("  >> new best saved")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

    np.save(os.path.join(RESULTS, "gru_v2_training_history.npy"), np.asarray(history))
    print(f"\nBest val loss: {best_val:.6f}")
    print("Saved:", best_path)


if __name__ == "__main__":
    main()
