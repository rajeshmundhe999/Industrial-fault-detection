"""Evaluate the augmented-trained TEP model. Same as evaluate_tep.py but uses
the aug checkpoint and saves to tep/results/aug/."""
import csv, os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

TEP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE    = os.path.dirname(TEP_DIR)
DATA    = os.path.join(TEP_DIR, "processed")
SAVED   = os.path.join(TEP_DIR, "saved", "gru_ae_v2_tep_aug_best.pt")
OUT     = os.path.join(TEP_DIR, "results", "aug")
sys.path.insert(0, BASE)
os.makedirs(OUT, exist_ok=True)
from model.gru_ae_v2 import GRUAutoencoderV2

SKIP_FAULTS = {3, 9}
FAULT_IDS   = [0] + [k for k in range(1, 22) if k not in SKIP_FAULTS]

def reconstruction_errors(model, x, device):
    model.eval(); chunks = []
    with torch.no_grad():
        for s in range(0, len(x), 512):
            z = torch.tensor(x[s:s+512], dtype=torch.float32).to(device)
            chunks.append(((model(z)-z)**2).cpu().numpy())
    return np.concatenate(chunks)

def robust_max_score(errors, median, mad):
    end_err = errors[:, -1, :]
    z = (end_err - median) / (mad + 1e-12)
    return z.max(axis=1)

def main():
    print("="*60); print("TEP AUGMENTED MODEL EVALUATION"); print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train   = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    X_val     = np.load(os.path.join(DATA, "X_val.npy")).astype(np.float32)
    X_test    = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)
    y_test    = np.load(os.path.join(DATA, "y_test.npy"))
    batch_ids = np.load(os.path.join(DATA, "batch_ids_test.npy"))
    print(f"X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")

    ckpt  = torch.load(SAVED, map_location="cpu", weights_only=False)
    model = GRUAutoencoderV2(input_size=22, dropout=0.2)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss',0):.6f}")

    print("Computing errors...")
    etr = reconstruction_errors(model, X_train, device)
    eva = reconstruction_errors(model, X_val,   device)
    ete = reconstruction_errors(model, X_test,  device)

    train_end = etr[:, -1, :]
    median    = np.median(train_end, axis=0)
    mad       = np.median(np.abs(train_end - median), axis=0) * 1.4826 + 1e-12

    val_scores  = robust_max_score(eva, median, mad)
    test_scores = robust_max_score(ete, median, mad)
    threshold   = float(val_scores.max())
    pred        = (test_scores > threshold).astype(int)

    roc_auc = roc_auc_score(y_test, test_scores)
    ap      = average_precision_score(y_test, test_scores)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)

    print(f"\nOVERALL:  AUC={roc_auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  "
          f"Recall={rec:.4f}  Precision={prec:.4f}")
    print(f"Threshold: {threshold:.4f}")

    fault_rows = []
    print("\nPER-FAULT:")
    for bid in np.unique(batch_ids):
        fid  = FAULT_IDS[bid]
        ix   = np.flatnonzero(batch_ids == bid)
        y_b  = y_test[ix]; sc_b = test_scores[ix]; pr_b = pred[ix]
        total = int(y_b.sum())
        det   = int(pr_b[y_b==1].sum())
        dr    = det/total if total > 0 else float("nan")
        auc_b = roc_auc_score(y_b, sc_b) if len(np.unique(y_b))==2 else float("nan")
        tag   = f"F{fid:02d}" if fid != 0 else "Normal"
        print(f"  batch={bid:2d}  {tag}  DR={dr:.3f}  AUC={auc_b:.4f}" if not np.isnan(auc_b)
              else f"  batch={bid:2d}  {tag}  DR={dr}")
        fault_rows.append(dict(batch_id=int(bid), fault_id=fid,
                               detection_rate=round(dr,4) if not np.isnan(dr) else "nan",
                               roc_auc=round(auc_b,4) if not np.isnan(auc_b) else "nan"))

    summary = dict(roc_auc=round(roc_auc,4), avg_precision=round(ap,4),
                   f1=round(f1,4), recall=round(rec,4), precision=round(prec,4),
                   threshold=round(threshold,4))
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section","metric","value"])
        for k,v in summary.items(): w.writerow(["overall",k,v])
        w.writerow([])
        w2 = csv.DictWriter(f, fieldnames=["section"]+list(fault_rows[0].keys()))
        w2.writeheader()
        for row in fault_rows: w2.writerow({"section":"per_fault",**row})
    np.save(os.path.join(OUT,"test_scores.npy"), test_scores)
    print(f"\nSaved to {OUT}")

if __name__ == "__main__":
    main()
