"""
Train GRU-VAE on IndPenSim normal data.

Key training choices:
  - β-annealing: start β=0 (pure reconstruction), linearly ramp to β_max=0.0005
    over BETA_WARMUP epochs. Prevents posterior collapse.
  - Monitor val ELBO (recon + beta*KL) for early stopping.
  - Same noise augmentation, AdamW, CosineAnnealingWarmRestarts as v2.
  - Input is MinMax-scaled data (same scaler as v2).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
SAVED = os.path.join(BASE, "model", "saved")
OUT   = os.path.join(BASE, "model", "results", "vae")
sys.path.insert(0, BASE)
from model.gru_vae import GRUVAEModel

os.makedirs(SAVED, exist_ok=True)
os.makedirs(OUT,   exist_ok=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS       = 500
BATCH_SIZE   = 256
LR           = 5e-4
PATIENCE     = 50
NOISE_SIGMA  = 0.01
LATENT_DIM   = 16
HIDDEN       = 64
DROPOUT      = 0.15
BETA_MAX     = 0.0005   # max KL weight — small keeps reconstruction sharp
BETA_WARMUP  = 100      # epochs to ramp β from 0 → BETA_MAX
BEST_PATH    = os.path.join(SAVED, "gru_vae_best.pt")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_noise(x, sigma=NOISE_SIGMA):
    return x + sigma * torch.randn_like(x)


def get_beta(epoch, warmup=BETA_WARMUP, beta_max=BETA_MAX):
    """Linear β warm-up schedule."""
    return min(beta_max, beta_max * epoch / warmup)


def main():
    print("=" * 60)
    print("GRU-VAE TRAINING — IndPenSim")
    print("=" * 60)
    print(f"Device : {DEVICE}")
    print(f"β_max  : {BETA_MAX}  warm-up: {BETA_WARMUP} epochs")

    # ── data ─────────────────────────────────────────────────────────────────
    X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(DATA, "X_val.npy")).astype(np.float32)
    print(f"Train : {X_train.shape}   Val : {X_val.shape}")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train)),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(X_val)),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── model ────────────────────────────────────────────────────────────────
    model = GRUVAEModel(input_size=9, hidden=HIDDEN,
                        latent_dim=LATENT_DIM, dropout=DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val   = float("inf")
    no_improve = 0
    history    = []
    t0         = time.time()

    for epoch in range(1, EPOCHS + 1):
        beta = get_beta(epoch)

        # ── train ────────────────────────────────────────────────────────────
        model.train()
        tr_elbo = tr_recon = tr_kl = 0.0
        for (x,) in train_dl:
            x  = x.to(DEVICE)
            xn = add_noise(x)
            recon, mu, logv = model(xn)
            loss, rc, kl    = model.elbo_loss(x, recon, mu, logv, beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = x.size(0)
            tr_elbo  += loss.item() * n
            tr_recon += rc * n
            tr_kl    += kl * n
        N = len(train_dl.dataset)
        tr_elbo /= N; tr_recon /= N; tr_kl /= N

        # ── validate ─────────────────────────────────────────────────────────
        model.eval()
        va_elbo = va_recon = va_kl = 0.0
        with torch.no_grad():
            for (x,) in val_dl:
                x = x.to(DEVICE)
                recon, mu, logv = model(x)
                loss, rc, kl    = model.elbo_loss(x, recon, mu, logv, beta)
                n = x.size(0)
                va_elbo  += loss.item() * n
                va_recon += rc * n
                va_kl    += kl * n
        Nv = len(val_dl.dataset)
        va_elbo /= Nv; va_recon /= Nv; va_kl /= Nv

        scheduler.step()
        history.append([epoch, tr_elbo, tr_recon, tr_kl, va_elbo, va_recon, va_kl, beta])

        if epoch % 20 == 0 or epoch <= 5:
            elapsed = time.time() - t0
            print(f"Ep {epoch:04d}/{EPOCHS}  "
                  f"tr_elbo={tr_elbo:.6f}  tr_recon={tr_recon:.6f}  tr_kl={tr_kl:.4f}  "
                  f"va_elbo={va_elbo:.6f}  va_recon={va_recon:.6f}  "
                  f"beta={beta:.5f}  {elapsed:.0f}s")

        # ── early stopping on val reconstruction (not ELBO) during warm-up,
        #    then on ELBO after warm-up ────────────────────────────────────
        monitor = va_recon if epoch <= BETA_WARMUP else va_elbo
        if monitor < best_val:
            best_val   = monitor
            no_improve = 0
            torch.save({
                "model_state_dict" : model.state_dict(),
                "epoch"            : epoch,
                "val_elbo"         : va_elbo,
                "val_recon"        : va_recon,
                "val_kl"           : va_kl,
                "beta"             : beta,
                "input_size"       : 9,
                "latent_dim"       : LATENT_DIM,
                "hidden"           : HIDDEN,
            }, BEST_PATH)
            if epoch % 20 == 0 or epoch <= 5:
                print(f"  >> best saved  (monitor={monitor:.6f})")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}  (patience={PATIENCE})")
            break

    np.save(os.path.join(OUT, "training_history.npy"), np.asarray(history))
    print(f"\nBest val monitor : {best_val:.6f}")
    print(f"Saved            : {BEST_PATH}")
    print(f"Total time       : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
