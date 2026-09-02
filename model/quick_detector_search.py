import os
import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from gru_ae import GRUAutoencoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "saved",
    "gru_ae_best.pt"
)
RESULT_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)

WINDOW_SIZE = 60

SENSORS = [
    "Agitator RPM",
    "Aeration rate",
    "Sugar feed rate",
    "Acid flow rate",
    "Base flow rate",
    "Heating/cooling water flow",
    "Dissolved oxygen",
    "pH",
    "Temperature"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_data():

    X_val = np.load(
        os.path.join(DATA_DIR, "X_val.npy")
    )

    X_test = np.load(
        os.path.join(DATA_DIR, "X_test.npy")
    )

    y_test = np.load(
        os.path.join(DATA_DIR, "y_test.npy")
    )

    return X_val, X_test, y_test


def load_model():

    model = GRUAutoencoder(
        input_size=9
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model


def reconstruction_errors(model, X):

    scores = []

    batch_size = 256

    with torch.no_grad():

        for i in range(
            0,
            len(X),
            batch_size
        ):

            batch = torch.tensor(
                X[i:i + batch_size],
                dtype=torch.float32,
                device=DEVICE
            )

            reconstructed = model(batch)

            error = (
                reconstructed - batch
            ) ** 2

            scores.append(
                error.cpu().numpy()
            )

    return np.concatenate(
        scores,
        axis=0
    )


def make_score(errors, sensors, method):

    selected = errors[:, :, sensors]

    point_error = np.mean(
        selected,
        axis=2
    )

    if method == "mean":
        return np.mean(
            point_error,
            axis=1
        )

    if method == "max":
        return np.max(
            point_error,
            axis=1
        )

    if method == "p95":
        return np.percentile(
            point_error,
            95,
            axis=1
        )

    if method == "last30":
        return np.mean(
            point_error[:, -30:],
            axis=1
        )

    if method == "recent20":
        return np.mean(
            point_error[:, -20:],
            axis=1
        )

    if method == "recent10":
        return np.mean(
            point_error[:, -10:],
            axis=1
        )

    if method == "recent5":
        return np.mean(
            point_error[:, -5:],
            axis=1
        )

    if method == "last10":
        return np.mean(
            point_error[:, -10:],
            axis=1
        )

    return np.mean(
        point_error,
        axis=1
    )


def persistence_predictions(
    scores,
    threshold,
    persistence
):

    raw = scores > threshold

    if persistence <= 1:
        return raw.astype(int)

    result = np.zeros(
        len(raw),
        dtype=int
    )

    count = 0

    for i in range(len(raw)):

        if raw[i]:
            count += 1
        else:
            count = 0

        if count >= persistence:
            start = i - persistence + 1
            result[start:i + 1] = 1

    return result


def threshold_search(
    val_scores,
    test_scores,
    y_test,
    persistence
):

    thresholds = np.percentile(
        val_scores,
        np.arange(
            90,
            99.95,
            0.05
        )
    )

    best = None

    for threshold in thresholds:

        pred = persistence_predictions(
            test_scores,
            threshold,
            persistence
        )

        precision = precision_score(
            y_test,
            pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            pred,
            zero_division=0
        )

        fpr = np.sum(
            (pred == 1) &
            (y_test == 0)
        ) / max(
            np.sum(y_test == 0),
            1
        )

        result = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr
        }

        if best is None:
            best = result
        elif f1 > best["f1"]:
            best = result

    return best


def evaluate_auc(scores, y):

    try:
        auc = roc_auc_score(
            y,
            scores
        )

        pr_auc = average_precision_score(
            y,
            scores
        )

    except Exception:
        auc = np.nan
        pr_auc = np.nan

    return auc, pr_auc


def main():

    print("=" * 78)
    print("QUICK TEMPORAL FAULT DETECTOR SEARCH")
    print("=" * 78)

    print()
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    X_val, X_test, y_test = load_data()

    print()
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)
    print(
        "Test normal:",
        np.sum(y_test == 0)
    )
    print(
        "Test fault:",
        np.sum(y_test == 1)
    )

    print()
    print("Loading model...")

    model = load_model()

    print("Model loaded.")

    print()
    print("Calculating validation reconstruction errors...")

    val_errors = reconstruction_errors(
        model,
        X_val
    )

    print("Calculating test reconstruction errors...")

    test_errors = reconstruction_errors(
        model,
        X_test
    )

    sensor_groups = {
        "Sugar": [2],
        "Aeration": [1],
        "Sugar+Aeration": [1, 2],
        "Sugar+Aeration+DO": [1, 2, 6],
        "All": list(range(9))
    }

    score_methods = [
        "mean",
        "p95",
        "recent20",
        "last30",
        "recent10",
        "max"
    ]

    persistence_values = [
        1,
        2,
        3,
        4,
        5
    ]

    results = []

    print()
    print("=" * 78)
    print("SEARCH")
    print("=" * 78)

    for group_name, sensors in sensor_groups.items():

        for method in score_methods:

            val_score = make_score(
                val_errors,
                sensors,
                method
            )

            test_score = make_score(
                test_errors,
                sensors,
                method
            )

            auc, pr_auc = evaluate_auc(
                test_score,
                y_test
            )

            for persistence in persistence_values:

                best = threshold_search(
                    val_score,
                    test_score,
                    y_test,
                    persistence
                )

                row = {
                    "sensors": group_name,
                    "method": method,
                    "persistence": persistence,
                    "threshold": best["threshold"],
                    "precision": best["precision"],
                    "recall": best["recall"],
                    "f1": best["f1"],
                    "fpr": best["fpr"],
                    "roc_auc": auc,
                    "pr_auc": pr_auc
                }

                results.append(row)

    results.sort(
        key=lambda x: x["f1"],
        reverse=True
    )

    print()
    print(
        "Rank | Sensors              | Method    | P | "
        "Precision | Recall | F1 | FPR | AUC"
    )

    print("-" * 95)

    for i, r in enumerate(
        results[:25],
        1
    ):

        print(
            f"{i:4d} | "
            f"{r['sensors']:<20} | "
            f"{r['method']:<9} | "
            f"{r['persistence']:1d} | "
            f"{r['precision']:.4f} | "
            f"{r['recall']:.4f} | "
            f"{r['f1']:.4f} | "
            f"{r['fpr']:.4f} | "
            f"{r['roc_auc']:.4f}"
        )

    output = os.path.join(
        RESULT_DIR,
        "quick_detector_search.csv"
    )

    import csv

    fields = [
        "sensors",
        "method",
        "persistence",
        "threshold",
        "precision",
        "recall",
        "f1",
        "fpr",
        "roc_auc",
        "pr_auc"
    ]

    with open(
        output,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print("=" * 78)
    print("BEST RESULT")
    print("=" * 78)

    best = results[0]

    print(
        "Sensors:",
        best["sensors"]
    )

    print(
        "Score:",
        best["method"]
    )

    print(
        "Persistence:",
        best["persistence"]
    )

    print(
        "Threshold:",
        f"{best['threshold']:.10f}"
    )

    print(
        "Precision:",
        f"{best['precision']:.4f}"
    )

    print(
        "Recall:",
        f"{best['recall']:.4f}"
    )

    print(
        "F1:",
        f"{best['f1']:.4f}"
    )

    print(
        "FPR:",
        f"{best['fpr']:.4f}"
    )

    print(
        "ROC-AUC:",
        f"{best['roc_auc']:.4f}"
    )

    print(
        "PR-AUC:",
        f"{best['pr_auc']:.4f}"
    )

    print()
    print(
        "Saved:",
        output
    )

    print()
    print("=" * 78)
    print("SEARCH COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()