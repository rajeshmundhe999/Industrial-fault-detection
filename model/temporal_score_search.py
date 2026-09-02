import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from gru_ae import GRUAutoencoder


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_TEST = os.path.join(BASE, "data", "processed", "X_test.npy")
Y_TEST = os.path.join(BASE, "data", "processed", "y_test.npy")
MODEL = os.path.join(BASE, "model", "saved", "gru_ae_best.pt")


def main():

    print("=" * 70)
    print("TEMPORAL GRU SCORE SEARCH")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    X = np.load(X_TEST)
    y = np.load(Y_TEST)

    model = GRUAutoencoder(
        input_size=9
    ).to(device)

    checkpoint = torch.load(
        MODEL,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    errors = []

    with torch.no_grad():

        for start in range(0, len(X), 256):

            end = min(start + 256, len(X))

            x = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            recon = model(x)

            e = (x - recon) ** 2

            errors.append(
                e.cpu().numpy()
            )

    errors = np.concatenate(errors)

    print()
    print("Error shape:", errors.shape)

    temporal = {}

    temporal["mean"] = errors.mean(axis=(1, 2))

    temporal["max"] = errors.max(axis=(1, 2))

    temporal["p95"] = np.percentile(
        errors,
        95,
        axis=(1, 2)
    )

    temporal["recent5"] = errors[:, -5:, :].mean(
        axis=(1, 2)
    )

    temporal["recent10"] = errors[:, -10:, :].mean(
        axis=(1, 2)
    )

    temporal["recent20"] = errors[:, -20:, :].mean(
        axis=(1, 2)
    )

    temporal["first30"] = errors[:, :30, :].mean(
        axis=(1, 2)
    )

    temporal["last30"] = errors[:, 30:, :].mean(
        axis=(1, 2)
    )

    temporal["temporal_ratio"] = (
        temporal["last30"] /
        (temporal["first30"] + 1e-12)
    )

    print()
    print("=" * 70)
    print("INDIVIDUAL SCORE PERFORMANCE")
    print("=" * 70)

    for name, score in temporal.items():

        roc = roc_auc_score(
            y,
            score
        )

        pr = average_precision_score(
            y,
            score
        )

        print(
            f"{name:<20} "
            f"ROC-AUC: {roc:.4f} | "
            f"PR-AUC: {pr:.4f}"
        )

    print()
    print("=" * 70)
    print("SCORE CORRELATIONS")
    print("=" * 70)

    names = list(temporal.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            a = temporal[names[i]]
            b = temporal[names[j]]

            corr = np.corrcoef(a, b)[0, 1]

            print(
                f"{names[i]:<18} vs "
                f"{names[j]:<18} "
                f"{corr:.4f}"
            )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()