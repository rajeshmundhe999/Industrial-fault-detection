import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from gru_ae import GRUAutoencoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "X_test.npy"
)

TEST_LABELS = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "y_test.npy"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "saved",
    "gru_ae_best.pt"
)


def calculate_errors(model, X, device):

    model.eval()

    errors = []

    batch_size = 256

    with torch.no_grad():

        for start in range(0, len(X), batch_size):

            end = min(
                start + batch_size,
                len(X)
            )

            x = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            output = model(x)

            error = (x - output) ** 2

            error = error.mean(dim=2)

            window_error = error.mean(dim=1)

            errors.append(
                window_error.cpu().numpy()
            )

    return np.concatenate(errors)


def describe(name, values):

    print(
        f"{name:<25} "
        f"Mean: {np.mean(values):.8f} | "
        f"Median: {np.median(values):.8f} | "
        f"95%: {np.percentile(values,95):.8f} | "
        f"Max: {np.max(values):.8f}"
    )


def main():

    print("=" * 70)
    print("GRU-AE RECONSTRUCTION ERROR ANALYSIS")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    X = np.load(TEST_DATA)

    y = np.load(TEST_LABELS)

    print()
    print("Test shape:", X.shape)
    print("Labels:", y.shape)

    print(
        "Normal:",
        np.sum(y == 0)
    )

    print(
        "Fault:",
        np.sum(y == 1)
    )

    model = GRUAutoencoder(
        input_size=9
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    print()
    print("Model loaded.")

    errors = calculate_errors(
        model,
        X,
        device
    )

    normal = errors[y == 0]
    fault = errors[y == 1]

    print()
    print("=" * 70)
    print("ERROR DISTRIBUTION")
    print("=" * 70)

    describe(
        "Normal",
        normal
    )

    describe(
        "Fault",
        fault
    )

    print()
    print("=" * 70)
    print("SEPARATION")
    print("=" * 70)

    print(
        "Mean ratio:",
        np.mean(fault) / np.mean(normal)
    )

    print(
        "Median ratio:",
        np.median(fault) / np.median(normal)
    )

    print(
        "Fault above normal 95th:",
        np.mean(
            fault >
            np.percentile(normal, 95)
        )
    )

    print(
        "Fault above normal 99th:",
        np.mean(
            fault >
            np.percentile(normal, 99)
        )
    )

    print()
    print("=" * 70)
    print("ROC / PR PERFORMANCE")
    print("=" * 70)

    roc = roc_auc_score(
        y,
        errors
    )

    pr = average_precision_score(
        y,
        errors
    )

    print(
        f"ROC-AUC: {roc:.6f}"
    )

    print(
        f"PR-AUC : {pr:.6f}"
    )

    print()
    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    percentiles = [
        95,
        97,
        98,
        98.5,
        99,
        99.5,
        99.7,
        99.8,
        99.9
    ]

    for p in percentiles:

        threshold = np.percentile(
            normal,
            p
        )

        predicted = (
            errors >= threshold
        )

        tp = np.sum(
            predicted & (y == 1)
        )

        fp = np.sum(
            predicted & (y == 0)
        )

        fn = np.sum(
            (~predicted) & (y == 1)
        )

        tn = np.sum(
            (~predicted) & (y == 0)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall > 0
            else 0
        )

        fpr = (
            fp / (fp + tn)
            if fp + tn > 0
            else 0
        )

        print(
            f"{p:5.1f}% | "
            f"{threshold:.8f} | "
            f"Precision {precision:.4f} | "
            f"Recall {recall:.4f} | "
            f"F1 {f1:.4f} | "
            f"FPR {fpr:.4f}"
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()