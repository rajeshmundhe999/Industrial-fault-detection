"""
Train and Evaluate LSTM-AE Baseline — leakage-safe protocol.
Mirrors the exact same training and evaluation setup as GRU-AE v2
so results are directly comparable.
"""
import csv, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_fscore_support)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data", "processed")
SAVED = os.path.join(BASE, "model", "saved")
OUT   = os.path.join(BASE, "model", "results", "lstm_ae")
sys.path.insert(0, BASE)
from model.lstm_ae import LSTMAEModel

os.makedirs(SAVED, exist_ok=True)
os.makedirs(OUT,   exist_ok=True)

EPOCHS      = 300
BATCH_SIZE  = 256
LR          = 1e-3
PATIENCE    = 40
NOISE_SIGMA = 0.01
WINDOW      = 60
STRIDE      = 5
BEST_PATH   = os.path.join(SAVED, "lstm_ae_best.pt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SENSORS     = ["Agitator RPM","Aeration rate","Sugar feed rate","Acid flow rate",
               "Base flow rate","Heating/cooling water flow","Dissolved oxygen","pH","Temperature"]


def scale(x, mn, sc): return x * sc + mn

def make_split(batch_data, faults, batches, mn, sc):
    xs, y_end, batch_ids, times = [], [], [], []
    for b in batches:
        x = scale(batch_data[str(b)], mn, sc)
        f = faults[str(b)]["fault"]
        t = faults[str(b)]["time"]
        for start in range(0, len(x) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            xs.append(x[start:end].astype(np.float32))
            y_end.append(int(f[end-1] > 0))
            batch_ids.append(b); times.append(float(t[end-1]))
    return (np.asarray(xs), np.asarray(y_end),
            np.asarray(batch_ids), np.asarray(times))

def get_errors(model, x_np, device, bs=512):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x_np), bs):
            xb = torch.tensor(x_np[i:i+bs], dtype=torch.float32).to(device)
            chunks.append(((model(xb) - xb) ** 2).cpu().numpy())
    return np.concatenate(chunks)  # (N, T, S)

def robust_max_score(errors, median, mad):
    end_err = errors[:, -1, :]
    z       = (end_err - median) / (mad + 1e-12)
    return z.max(axis=1)

def event_recall(pred, batch_ids, faulty):
    return sum(1 for b in faulty if pred[batch_ids == b].any()) / len(faulty)

def compute_metrics(y, pred, score, batch_ids, faulty):
    roc = roc_auc_score(y, score)
    ap  = average_precision_score(y, score)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    er  = event_recall(pred, batch_ids, faulty)
    return dict(roc_auc=round(roc,4), avg_prec=round(ap,4),
                f1=round(f,4), recall=round(r,4), precision=round(p,4),
                event_recall=round(er,4))


def main():
    print("=" * 60)
    print("LSTM-AE BASELINE — Train + Evaluate")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── data ─────────────────────────────────────────────────────────────────
    X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    X_val   = np.load(os.path.join(DATA, "X_val.npy")).astype(np.float32)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train)),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(X_val)),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── train ─────────────────────────────────────────────────────────────────
    model = LSTMAEModel(input_size=9, hidden=32, latent_dim=8, dropout=0.1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

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
            xn    = x + NOISE_SIGMA * torch.randn_like(x)
            loss  = criterion(model(xn), x)
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

        if epoch % 50 == 0 or epoch <= 3:
            print(f"Ep {epoch:03d}/{EPOCHS}  train={tr_loss:.6f}  "
                  f"val={va_loss:.6f}  [{(time.time()-t0)/60:.1f}min]")

        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss": va_loss, "epoch": epoch}, BEST_PATH)
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    print(f"Best val loss: {best_val:.6f}  [{(time.time()-t0)/60:.1f}min]")

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\n--- Evaluation (leakage-safe) ---")
    ckpt = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)

    batch_data = np.load(os.path.join(DATA, "batch_data.npy"), allow_pickle=True).item()
    faults     = np.load(os.path.join(DATA, "batch_faults.npy"), allow_pickle=True).item()
    sc_mn      = np.load(os.path.join(DATA, "scaler_min.npy")).astype(np.float32)
    sc_sc      = np.load(os.path.join(DATA, "scaler_scale.npy")).astype(np.float32)

    faulty_batches = list(range(91, 101))
    xtr, _,   _,   _   = make_split(batch_data, faults, list(range(1,71)),  sc_mn, sc_sc)
    xva, _,   _,   _   = make_split(batch_data, faults, list(range(71,81)), sc_mn, sc_sc)
    xte, yte, bte, tte = make_split(batch_data, faults, list(range(81,101)),sc_mn, sc_sc)

    etr = get_errors(model, xtr, DEVICE)
    eva = get_errors(model, xva, DEVICE)
    ete = get_errors(model, xte, DEVICE)

    train_end = etr[:, -1, :]
    median    = np.median(train_end, axis=0)
    mad       = np.median(np.abs(train_end - median), axis=0) * 1.4826 + 1e-12

    val_scores  = robust_max_score(eva, median, mad)
    test_scores = robust_max_score(ete, median, mad)
    threshold   = float(val_scores.max())
    pred        = (test_scores > threshold).astype(int)

    m = compute_metrics(yte, pred, test_scores, bte, faulty_batches)
    print(f"LSTM-AE: AUC={m['roc_auc']}  F1={m['f1']}  "
          f"Recall={m['recall']}  Precision={m['precision']}  "
          f"EvRec={m['event_recall']}")

    # ── save ──────────────────────────────────────────────────────────────────
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in m.items():
            w.writerow([k, v])
        w.writerow(["threshold", round(threshold, 4)])
        w.writerow(["n_params", n_params])

    np.save(os.path.join(OUT, "test_scores.npy"),  test_scores)
    np.save(os.path.join(OUT, "val_scores.npy"),   val_scores)
    np.save(os.path.join(OUT, "test_labels.npy"),  yte)
    np.save(os.path.join(OUT, "test_batch_ids.npy"), bte)
    np.save(os.path.join(OUT, "training_history.npy"), np.asarray(history))

    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
