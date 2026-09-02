import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULT_DIR = os.path.join(BASE_DIR, "model", "results")
SAVED_DIR = os.path.join(BASE_DIR, "model", "saved")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(SAVED_DIR, "gru_temporal_best.pt")
RESULT_PATH = os.path.join(RESULT_DIR, "temporal_fusion_results.csv")


class TemporalGRU(nn.Module):

    def __init__(self, input_size=9, hidden_size=64, num_layers=2):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size * 3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):

        first = torch.zeros_like(x)

        first[:, 1:, :] = (
            x[:, 1:, :]
            -
            x[:, :-1, :]
        )

        second = torch.zeros_like(x)

        second[:, 2:, :] = (
            first[:, 2:, :]
            -
            first[:, 1:-1, :]
        )

        augmented = torch.cat(
            [x, first, second],
            dim=2
        )

        encoded, _ = self.gru(
            augmented
        )

        output = self.decoder(
            encoded
        )

        return output


def derivatives(X):

    first = np.zeros_like(X)

    first[:, 1:, :] = (
        X[:, 1:, :]
        -
        X[:, :-1, :]
    )

    second = np.zeros_like(X)

    second[:, 2:, :] = (
        first[:, 2:, :]
        -
        first[:, 1:-1, :]
    )

    return first, second


def temporal_score(
    X,
    first_mean,
    first_std,
    second_mean,
    second_std
):

    first, second = derivatives(X)

    first_z = np.abs(
        first - first_mean
    ) / (
        first_std + 1e-8
    )

    second_z = np.abs(
        second - second_mean
    ) / (
        second_std + 1e-8
    )

    first_score = np.mean(
        first_z,
        axis=(1, 2)
    )

    second_score = np.mean(
        second_z,
        axis=(1, 2)
    )

    return (
        0.7 * first_score
        +
        0.3 * second_score
    )


def reconstruction_scores(
    model,
    X,
    batch_size=128
):

    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X,
                dtype=torch.float32
            )
        ),
        batch_size=batch_size,
        shuffle=False
    )

    scores = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            x = batch[0].to(
                DEVICE
            )

            reconstruction = model(x)

            error = (
                reconstruction - x
            ) ** 2

            score = torch.mean(
                error,
                dim=(1, 2)
            )

            scores.append(
                score.cpu().numpy()
            )

    return np.concatenate(scores)


def evaluate(
    labels,
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    auc = roc_auc_score(
        labels,
        scores
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (
        fp + tn
    )

    return (
        precision,
        recall,
        f1,
        accuracy,
        auc,
        fpr,
        cm
    )


def train(
    model,
    X_train,
    X_val
):

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_train,
                dtype=torch.float32
            )
        ),
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_val,
                dtype=torch.float32
            )
        ),
        batch_size=128,
        shuffle=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8
    )

    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(1, 101):

        model.train()

        total = 0
        count = 0

        for batch in train_loader:

            x = batch[0].to(
                DEVICE
            )

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output,
                x
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total += (
                loss.item()
                *
                x.size(0)
            )

            count += x.size(0)

        train_loss = total / count

        model.eval()

        total = 0
        count = 0

        with torch.no_grad():

            for batch in val_loader:

                x = batch[0].to(
                    DEVICE
                )

                output = model(x)

                loss = criterion(
                    output,
                    x
                )

                total += (
                    loss.item()
                    *
                    x.size(0)
                )

                count += x.size(0)

        val_loss = total / count

        scheduler.step(
            val_loss
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/100 | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {lr:.2e}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "  New best model saved."
            )

    return best_loss


def main():

    print("=" * 75)
    print(
        "TEMPORAL-ENHANCED GRU-AE"
    )
    print("=" * 75)

    print(
        "\nDevice:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    X_train = np.load(
        os.path.join(
            DATA_DIR,
            "X_train.npy"
        )
    ).astype(np.float32)

    X_val = np.load(
        os.path.join(
            DATA_DIR,
            "X_val.npy"
        )
    ).astype(np.float32)

    X_test = np.load(
        os.path.join(
            DATA_DIR,
            "X_test.npy"
        )
    ).astype(np.float32)

    labels = np.load(
        os.path.join(
            RESULT_DIR,
            "test_labels.npy"
        )
    )

    batch_ids = np.load(
        os.path.join(
            RESULT_DIR,
            "test_batch_ids.npy"
        )
    )

    print(
        "\nTraining:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print(
        "\nCalculating training temporal statistics..."
    )

    train_first, train_second = derivatives(
        X_train
    )

    first_mean = np.mean(
        train_first,
        axis=(0, 1),
        keepdims=True
    )

    first_std = np.std(
        train_first,
        axis=(0, 1),
        keepdims=True
    )

    second_mean = np.mean(
        train_second,
        axis=(0, 1),
        keepdims=True
    )

    second_std = np.std(
        train_second,
        axis=(0, 1),
        keepdims=True
    )

    print(
        "\nTraining temporal model..."
    )

    model = TemporalGRU().to(
        DEVICE
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    best_loss = train(
        model,
        X_train,
        X_val
    )

    print(
        "\nBest validation loss:",
        best_loss
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    print(
        "\nCalculating reconstruction scores..."
    )

    train_recon = reconstruction_scores(
        model,
        X_train
    )

    val_recon = reconstruction_scores(
        model,
        X_val
    )

    test_recon = reconstruction_scores(
        model,
        X_test
    )

    print(
        "Calculating temporal scores..."
    )

    train_temp = temporal_score(
        X_train,
        first_mean,
        first_std,
        second_mean,
        second_std
    )

    val_temp = temporal_score(
        X_val,
        first_mean,
        first_std,
        second_mean,
        second_std
    )

    test_temp = temporal_score(
        X_test,
        first_mean,
        first_std,
        second_mean,
        second_std
    )

    def normalize(
        train,
        val,
        test
    ):

        low = np.percentile(
            train,
            1
        )

        high = np.percentile(
            train,
            99
        )

        scale = high - low

        if scale <= 0:

            scale = 1e-8

        return (
            np.clip(
                (train - low) / scale,
                0,
                1
            ),
            np.clip(
                (val - low) / scale,
                0,
                1
            ),
            np.clip(
                (test - low) / scale,
                0,
                1
            )
        )

    (
        train_r,
        val_r,
        test_r
    ) = normalize(
        train_recon,
        val_recon,
        test_recon
    )

    (
        train_t,
        val_t,
        test_t
    ) = normalize(
        train_temp,
        val_temp,
        test_temp
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "ABLATION / FUSION RESULTS"
    )

    print(
        "=" * 75
    )

    rows = []

    for weight in np.arange(
        0.0,
        1.01,
        0.05
    ):

        train_score = (
            weight * train_r
            +
            (1 - weight) * train_t
        )

        test_score = (
            weight * test_r
            +
            (1 - weight) * test_t
        )

        threshold = np.percentile(
            train_score,
            99.5
        )

        (
            precision,
            recall,
            f1,
            accuracy,
            auc,
            fpr,
            cm
        ) = evaluate(
            labels,
            test_score,
            threshold
        )

        rows.append(
            {
                "reconstruction_weight":
                    round(float(weight), 2),
                "temporal_weight":
                    round(float(1 - weight), 2),
                "threshold":
                    float(threshold),
                "precision":
                    float(precision),
                "recall":
                    float(recall),
                "f1":
                    float(f1),
                "accuracy":
                    float(accuracy),
                "roc_auc":
                    float(auc),
                "fpr":
                    float(fpr)
            }
        )

        print(
            f"Recon {weight:.2f} | "
            f"Temporal {1-weight:.2f} | "
            f"F1 {f1:.4f} | "
            f"AUC {auc:.4f} | "
            f"Precision {precision:.4f} | "
            f"Recall {recall:.4f} | "
            f"FPR {fpr:.4f}"
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        RESULT_PATH,
        index=False
    )

    best = df.loc[
        df["f1"].idxmax()
    ]

    print(
        "\n" + "=" * 75
    )

    print(
        "BEST TEMPORAL FUSION"
    )

    print(
        "=" * 75
    )

    print(
        "Reconstruction weight:",
        best["reconstruction_weight"]
    )

    print(
        "Temporal weight:",
        best["temporal_weight"]
    )

    print(
        "F1:",
        best["f1"]
    )

    print(
        "ROC-AUC:",
        best["roc_auc"]
    )

    print(
        "Precision:",
        best["precision"]
    )

    print(
        "Recall:",
        best["recall"]
    )

    print(
        "Accuracy:",
        best["accuracy"]
    )

    print(
        "FPR:",
        best["fpr"]
    )

    print(
        "\nResults saved:",
        RESULT_PATH
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "TEMPORAL EXPERIMENT COMPLETE"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()