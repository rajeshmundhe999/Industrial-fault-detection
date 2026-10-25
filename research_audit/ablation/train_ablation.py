"""
Architecture Ablation Study for GRU-AE v2
==========================================
Trains 3 ablation variants against the full v2 model using the same
leakage-safe protocol (train batches 1-70, val 71-80, test 81-100).

Variants:
  A) No Attention  — replace AttentionPool with identity (use raw GRU output)
  B) Unidirectional — replace BiGRU encoder with forward-only GRU
  C) Shared Head   — replace per-sensor heads with single shared linear head
  D) Full v2       — the original model (run evaluation only, not retrain)

Usage:
  py -3.12 research_audit/ablation/train_ablation.py

Results saved to: research_audit/ablation/results/
"""
import os, sys, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data", "processed")
RESULTS_DIR = os.path.join(BASE, "research_audit", "ablation", "results")
sys.path.insert(0, BASE)

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EPOCHS = 400
BATCH_SIZE = 256
LR = 5e-4
PATIENCE = 50
NOISE_SIGMA = 0.01
SEED = 42

# ─── Ablation Model Definitions ──────────────────────────────────────────────

class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.scale = hidden ** -0.5

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ v


class AblationNoAttention(nn.Module):
    """Variant A: BiGRU encoder WITHOUT attention (just raw GRU output)."""

    def __init__(self, input_size=9, dropout=0.15):
        super().__init__()
        self.enc1 = nn.GRU(input_size, 32, batch_first=True, bidirectional=True)
        self.ln_enc1 = nn.LayerNorm(64)
        self.drop_enc = nn.Dropout(dropout)
        # NO attention — pass GRU output directly
        self.enc2 = nn.GRU(64, 16, batch_first=True)
        self.ln_enc2 = nn.LayerNorm(16)
        self.dec1 = nn.GRU(16, 32, batch_first=True)
        self.ln_dec1 = nn.LayerNorm(32)
        self.drop_dec = nn.Dropout(dropout)
        self.dec2 = nn.GRU(32, 64, batch_first=True)
        self.ln_dec2 = nn.LayerNorm(64)
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(input_size)])

    def encode(self, x):
        z, _ = self.enc1(x)
        z = self.ln_enc1(z)
        z = self.drop_enc(z)
        # Identity (no attention reweighting)
        z, _ = self.enc2(z)
        z = self.ln_enc2(z)
        return z

    def decode(self, z):
        h, _ = self.dec1(z)
        h = self.ln_dec1(h)
        h = self.drop_dec(h)
        h, _ = self.dec2(h)
        h = self.ln_dec2(h)
        return torch.cat([head(h) for head in self.heads], dim=-1)

    def forward(self, x):
        return self.decode(self.encode(x))


class AblationUnidirectional(nn.Module):
    """Variant B: Unidirectional GRU (no bidirectional encoder)."""

    def __init__(self, input_size=9, dropout=0.15):
        super().__init__()
        self.enc1 = nn.GRU(input_size, 64, batch_first=True, bidirectional=False)  # 64 to match param count
        self.ln_enc1 = nn.LayerNorm(64)
        self.drop_enc = nn.Dropout(dropout)
        self.attention = AttentionPool(64)
        self.enc2 = nn.GRU(64, 16, batch_first=True)
        self.ln_enc2 = nn.LayerNorm(16)
        self.dec1 = nn.GRU(16, 32, batch_first=True)
        self.ln_dec1 = nn.LayerNorm(32)
        self.drop_dec = nn.Dropout(dropout)
        self.dec2 = nn.GRU(32, 64, batch_first=True)
        self.ln_dec2 = nn.LayerNorm(64)
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(input_size)])

    def encode(self, x):
        z, _ = self.enc1(x)
        z = self.ln_enc1(z)
        z = self.drop_enc(z)
        z = self.attention(z)
        z, _ = self.enc2(z)
        z = self.ln_enc2(z)
        return z

    def decode(self, z):
        h, _ = self.dec1(z)
        h = self.ln_dec1(h)
        h = self.drop_dec(h)
        h, _ = self.dec2(h)
        h = self.ln_dec2(h)
        return torch.cat([head(h) for head in self.heads], dim=-1)

    def forward(self, x):
        return self.decode(self.encode(x))


class AblationSharedHead(nn.Module):
    """Variant C: Single shared output head (no per-sensor specialization)."""

    def __init__(self, input_size=9, dropout=0.15):
        super().__init__()
        self.input_size = input_size
        self.enc1 = nn.GRU(input_size, 32, batch_first=True, bidirectional=True)
        self.ln_enc1 = nn.LayerNorm(64)
        self.drop_enc = nn.Dropout(dropout)
        self.attention = AttentionPool(64)
        self.enc2 = nn.GRU(64, 16, batch_first=True)
        self.ln_enc2 = nn.LayerNorm(16)
        self.dec1 = nn.GRU(16, 32, batch_first=True)
        self.ln_dec1 = nn.LayerNorm(32)
        self.drop_dec = nn.Dropout(dropout)
        self.dec2 = nn.GRU(32, 64, batch_first=True)
        self.ln_dec2 = nn.LayerNorm(64)
        # SINGLE shared head instead of per-sensor heads
        self.shared_head = nn.Linear(64, input_size)

    def encode(self, x):
        z, _ = self.enc1(x)
        z = self.ln_enc1(z)
        z = self.drop_enc(z)
        z = self.attention(z)
        z, _ = self.enc2(z)
        z = self.ln_enc2(z)
        return z

    def decode(self, z):
        h, _ = self.dec1(z)
        h = self.ln_dec1(h)
        h = self.drop_dec(h)
        h, _ = self.dec2(h)
        h = self.ln_dec2(h)
        return self.shared_head(h)  # (B, T, n_sensors) directly

    def forward(self, x):
        return self.decode(self.encode(x))


# ─── Scoring ─────────────────────────────────────────────────────────────────

def compute_window_scores(model, X_tensor):
    """Compute per-window anomaly scores: mean sensor reconstruction error."""
    model.eval()
    all_scores = []
    dl = DataLoader(TensorDataset(X_tensor), batch_size=512, shuffle=False)
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            recon = model(xb)
            err = (xb - recon).abs()      # (B, T, n_sensors)
            # Per-sensor: take mean over time, then top-k max over sensors
            per_sensor = err.mean(dim=1)   # (B, n_sensors)
            score = per_sensor.max(dim=1).values  # (B,) — max sensor error
            all_scores.append(score.cpu().numpy())
    return np.concatenate(all_scores)


def evaluate_model(model, X_val, X_test, y_test, batch_ids_test, variant_name):
    """Compute threshold on val, evaluate on test."""
    val_tensor = torch.FloatTensor(X_val)
    test_tensor = torch.FloatTensor(X_test)

    val_scores = compute_window_scores(model, val_tensor)
    test_scores = compute_window_scores(model, test_tensor)

    threshold = val_scores.max()  # val_max — leakage-safe
    preds = (test_scores >= threshold).astype(int)

    auc = roc_auc_score(y_test, test_scores)
    ap = average_precision_score(y_test, test_scores)
    f1 = f1_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)

    # Event recall
    fault_batches = [b for b in np.unique(batch_ids_test) if y_test[batch_ids_test == b].any()]
    detected = sum(1 for b in fault_batches if preds[batch_ids_test == b].any())
    event_recall = detected / len(fault_batches) if fault_batches else 0.0

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "variant": variant_name,
        "n_params": n_params,
        "threshold": threshold,
        "auc": auc,
        "avg_precision": ap,
        "f1": f1,
        "recall": rec,
        "precision": prec,
        "event_recall": event_recall,
    }


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_model(model, X_train, X_val, variant_name, save_path):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = TensorDataset(torch.FloatTensor(X_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    best_val = float("inf")
    no_improve = 0
    history = []

    print(f"\n{'='*60}")
    print(f"TRAINING: {variant_name}  [{sum(p.numel() for p in model.parameters()):,} params]")
    print(f"{'='*60}")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for (x,) in train_dl:
            x = x.to(DEVICE)
            xn = x + NOISE_SIGMA * torch.randn_like(x)
            recon = model(xn)
            loss = criterion(recon, x)
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

        if epoch % 50 == 0 or epoch <= 5:
            elapsed = (time.time() - t0) / 60
            print(f"  Epoch {epoch:03d}/{EPOCHS}  train={tr_loss:.6f}  val={va_loss:.6f}  ({elapsed:.1f} min)")

        if va_loss < best_val:
            best_val = va_loss
            no_improve = 0
            torch.save({"state_dict": model.state_dict(), "val_loss": va_loss, "epoch": epoch}, save_path)
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    elapsed = (time.time() - t0) / 60
    print(f"  Done in {elapsed:.1f} min. Best val_loss={best_val:.6f}")
    np.save(save_path.replace(".pt", "_history.npy"), np.array(history))
    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    X_val = np.load(os.path.join(DATA, "X_val.npy")).astype(np.float32)
    X_test = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(DATA, "X_test.npy").replace("X_test", "y_test") if
                     os.path.exists(os.path.join(DATA, "y_test.npy")) else
                     os.path.join(BASE, "data", "processed", "y_test.npy")).astype(int)

    # Load y_test properly
    y_test_path = os.path.join(BASE, "data", "processed", "y_test.npy")
    batch_ids_path = os.path.join(BASE, "data", "processed", "batch_ids_test.npy")
    if os.path.exists(y_test_path):
        y_test = np.load(y_test_path).astype(int)
    else:
        # Reconstruct from leakage_safe_v2
        d = np.load(os.path.join(BASE, "model", "results", "leakage_safe_v2", "scores_and_predictions.npz"))
        y_test = d["y_end"].astype(int)

    if os.path.exists(batch_ids_path):
        batch_ids_test = np.load(batch_ids_path)
    else:
        d = np.load(os.path.join(BASE, "model", "results", "leakage_safe_v2", "scores_and_predictions.npz"))
        batch_ids_test = d["batch_ids"]

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"y_test fault fraction: {y_test.mean():.4f}")

    # Also evaluate the already-trained full v2 model
    from model.gru_ae_v2 import GRUAutoencoderV2
    full_model = GRUAutoencoderV2(input_size=9, dropout=0.15).to(DEVICE)
    ckpt = torch.load(os.path.join(BASE, "model", "saved", "gru_ae_v2_best.pt"),
                      map_location=DEVICE, weights_only=True)
    full_model.load_state_dict(ckpt["model_state_dict"])
    print("\nEvaluating existing Full v2 model...")
    full_result = evaluate_model(full_model, X_val, X_test, y_test, batch_ids_test, "Full_v2 (BiGRU+Attn+PerSensor)")
    print(f"  AUC={full_result['auc']:.4f}  F1={full_result['f1']:.4f}  EvtRec={full_result['event_recall']:.3f}")

    results = [full_result]

    # Train ablation variants
    variants = [
        ("A_NoAttention",    AblationNoAttention(input_size=9)),
        ("B_Unidirectional", AblationUnidirectional(input_size=9)),
        ("C_SharedHead",     AblationSharedHead(input_size=9)),
    ]

    for name, model in variants:
        model = model.to(DEVICE)
        save_path = os.path.join(RESULTS_DIR, f"{name}_best.pt")
        hist_path = save_path.replace(".pt", "_history.npy")

        if os.path.exists(save_path) and os.path.exists(hist_path):
            print(f"\n[RESUME] Found completed training artifacts for {name}. Skipping retraining.", flush=True)
        else:
            model = train_model(model, X_train, X_val, name, save_path)

        # Reload best checkpoint
        ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

        print(f"Evaluating {name}...")
        r = evaluate_model(model, X_val, X_test, y_test, batch_ids_test, name)
        results.append(r)
        print(f"  AUC={r['auc']:.4f}  F1={r['f1']:.4f}  EvtRec={r['event_recall']:.3f}")

    # Save results
    import csv
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<40} {'Params':>8} {'AUC':>7} {'F1':>7} {'Recall':>8} {'EvtRec':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['variant']:<40} {r['n_params']:>8,} {r['auc']:>7.4f} {r['f1']:>7.4f} "
              f"{r['recall']:>8.4f} {r['event_recall']:>8.3f}")
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
