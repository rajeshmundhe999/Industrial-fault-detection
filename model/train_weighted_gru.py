import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
SAVE_DIR = os.path.join(BASE_DIR, "saved")

sys.path.insert(0, BASE_DIR)

from gru_ae import GRUAutoencoder


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_SIZE = 9
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3

MODEL_PATH = os.path.join(
    SAVE_DIR,
    "weighted_gru_best.pt"
)

SENSOR_NAMES = [
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

SENSOR_WEIGHTS = torch.tensor(
    [
        0.50,
        1.50,
        2.00,
        0.75,
        1.25,
        1.25,
        1.75,
        1.00,
        0.75
    ],
    dtype=torch.float32
)

SENSOR_WEIGHTS = (
    SENSOR_WEIGHTS /
    SENSOR_WEIGHTS.mean()
)


def weighted_loss(
    reconstruction,
    target,
    weights
):

    error = (
        reconstruction -
        target
    ) ** 2

    error = error * weights.view(
        1,
        1,
        -1
    )

    return error.mean()


def evaluate(
    model,
    loader,
    weights
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for x, in loader:

            x = x.to(DEVICE)

            output = model(x)

            loss = weighted_loss(
                output,
                x,
                weights
            )

            batch_size = x.size(0)

            total_loss += (
                loss.item() *
                batch_size
            )

            total_samples += batch_size

    return (
        total_loss /
        total_samples
    )


def main():

    print("=" * 60)
    print("WEIGHTED GRU-AE TRAINING")
    print("=" * 60)

    print(
        "\nDevice:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory /
                (1024 ** 3),
                2
            ),
            "GB"
        )

    X_train = np.load(
        os.path.join(
            DATA_DIR,
            "X_train.npy"
        )
    ).astype(
        np.float32
    )

    X_val = np.load(
        os.path.join(
            DATA_DIR,
            "X_val.npy"
        )
    ).astype(
        np.float32
    )

    print(
        "\nTraining data:",
        X_train.shape
    )

    print(
        "Validation data:",
        X_val.shape
    )

    print(
        "\nSensor weights:"
    )

    for name, weight in zip(
        SENSOR_NAMES,
        SENSOR_WEIGHTS.numpy()
    ):

        print(
            f"{name:35s} {weight:.3f}"
        )

    train_dataset = TensorDataset(
        torch.from_numpy(
            X_train
        )
    )

    val_dataset = TensorDataset(
        torch.from_numpy(
            X_val
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    model = GRUAutoencoder(
        input_size=INPUT_SIZE
    ).to(
        DEVICE
    )

    print(
        "\nParameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    weights = SENSOR_WEIGHTS.to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7
    )

    best_val = float("inf")

    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )

    print(
        "\nStarting training..."
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        samples = 0

        for x, in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(x)

            loss = weighted_loss(
                output,
                x,
                weights
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            batch_size = x.size(0)

            running_loss += (
                loss.item() *
                batch_size
            )

            samples += batch_size

        train_loss = (
            running_loss /
            samples
        )

        val_loss = evaluate(
            model,
            val_loader,
            weights
        )

        scheduler.step(
            val_loss
        )

        lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {lr:.2e}"
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_size":
                        INPUT_SIZE,

                    "sensor_weights":
                        SENSOR_WEIGHTS.numpy(),

                    "best_val_loss":
                        best_val,

                    "epoch":
                        epoch
                },
                MODEL_PATH
            )

            print(
                "  New best model saved."
            )

    print(
        "\n" + "=" * 60
    )

    print(
        "WEIGHTED GRU-AE TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "Best validation loss:",
        best_val
    )

    print(
        "Model:",
        os.path.abspath(
            MODEL_PATH
        )
    )


if __name__ == "__main__":

    main()