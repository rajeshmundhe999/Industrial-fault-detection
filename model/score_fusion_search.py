import os
import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from gru_ae import GRUAutoencoder


BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

X_VAL = os.path.join(
    BASE,
    "data",
    "processed",
    "X_val.npy"
)

X_TEST = os.path.join(
    BASE,
    "data",
    "processed",
    "X_test.npy"
)

Y_TEST = os.path.join(
    BASE,
    "data",
    "processed",
    "y_test.npy"
)

MODEL = os.path.join(
    BASE,
    "model",
    "saved",
    "gru_ae_best.pt"
)


def calculate_errors(model, X, device):

    errors = []

    with torch.no_grad():

        for start in range(0, len(X), 256):

            end = min(
                start + 256,
                len(X)
            )

            x = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            recon = model(x)

            error = (
                x - recon
            ) ** 2

            errors.append(
                error.cpu().numpy()
            )

    return np.concatenate(errors)


def make_scores(errors):

    scores = {}

    scores["mean"] = errors.mean(
        axis=(1, 2)
    )

    scores["p95"] = np.percentile(
        errors,
        95,
        axis=(1, 2)
    )

    scores["recent20"] = errors[:, -20:, :].mean(
        axis=(1, 2)
    )

    scores["last30"] = errors[:, 30:, :].mean(
        axis=(1, 2)
    )

    scores["max"] = errors.max(
        axis=(1, 2)
    )

    return scores


def normalize(
    score,
    val_score
):

    lo = np.percentile(
        val_score,
        1
    )

    hi = np.percentile(
        val_score,
        99
    )

    return np.clip(
        (score - lo) /
        (hi - lo + 1e-12),
        0,
        10
    )


def evaluate(
    score,
    threshold,
    y
):

    pred = (
        score >= threshold
    ).astype(int)

    return (
        precision_score(
            y,
            pred,
            zero_division=0
        ),
        recall_score(
            y,
            pred,
            zero_division=0
        ),
        f1_score(
            y,
            pred,
            zero_division=0
        )
    )


def main():

    print("=" * 70)
    print("GRU RECONSTRUCTION SCORE FUSION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    X_val = np.load(
        X_VAL
    )

    X_test = np.load(
        X_TEST
    )

    y_test = np.load(
        Y_TEST
    )

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
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print()
    print("Calculating validation errors...")

    val_errors = calculate_errors(
        model,
        X_val,
        device
    )

    print("Calculating test errors...")

    test_errors = calculate_errors(
        model,
        X_test,
        device
    )

    val_scores = make_scores(
        val_errors
    )

    test_scores = make_scores(
        test_errors
    )

    normalized_val = {}
    normalized_test = {}

    for name in val_scores:

        normalized_val[name] = normalize(
            val_scores[name],
            val_scores[name]
        )

        normalized_test[name] = normalize(
            test_scores[name],
            val_scores[name]
        )

    print()
    print("=" * 70)
    print("SEARCHING SCORE COMBINATIONS")
    print("=" * 70)

    configurations = []

    for w_mean in np.arange(
        0.5,
        1.01,
        0.1
    ):

        for w_recent in np.arange(
            0,
            0.51,
            0.1
        ):

            for w_p95 in np.arange(
                0,
                0.51,
                0.1
            ):

                if (
                    w_mean +
                    w_recent +
                    w_p95
                    > 1.001
                ):
                    continue

                remaining = (
                    1
                    - w_mean
                    - w_recent
                    - w_p95
                )

                score_val = (
                    w_mean *
                    normalized_val["mean"]
                    +
                    w_recent *
                    normalized_val["recent20"]
                    +
                    w_p95 *
                    normalized_val["p95"]
                    +
                    remaining *
                    normalized_val["last30"]
                )

                score_test = (
                    w_mean *
                    normalized_test["mean"]
                    +
                    w_recent *
                    normalized_test["recent20"]
                    +
                    w_p95 *
                    normalized_test["p95"]
                    +
                    remaining *
                    normalized_test["last30"]
                )

                threshold = np.percentile(
                    score_val,
                    99.0
                )

                precision, recall, f1 = evaluate(
                    score_test,
                    threshold,
                    y_test
                )

                auc = roc_auc_score(
                    y_test,
                    score_test
                )

                configurations.append(
                    (
                        f"{w_mean:.1f}",
                        f"{w_recent:.1f}",
                        f"{w_p95:.1f}",
                        f"{remaining:.1f}",
                        precision,
                        recall,
                        f1,
                        auc
                    )
                )

    configurations.sort(
        key=lambda x: x[6],
        reverse=True
    )

    print()
    print(
        "Mean | Recent20 | P95 | Last30 | "
        "Precision | Recall | F1 | AUC"
    )

    print("-" * 95)

    for row in configurations[:15]:

        print(
            f"{row[0]:>4} | "
            f"{row[1]:>8} | "
            f"{row[2]:>3} | "
            f"{row[3]:>6} | "
            f"{row[4]:>9.4f} | "
            f"{row[5]:>6.4f} | "
            f"{row[6]:>6.4f} | "
            f"{row[7]:>6.4f}"
        )

    print()
    print("=" * 70)
    print("REFERENCE")
    print("=" * 70)

    print(
        "GRU mean ROC-AUC:",
        roc_auc_score(
            y_test,
            test_scores["mean"]
        )
    )

    print(
        "GRU mean PR-AUC:",
        average_precision_score(
            y_test,
            test_scores["mean"]
        )
    )

    print()
    print("=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()