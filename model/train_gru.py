import os
import sys
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.gru_ae import GRUAutoencoder


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "saved"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 60)
    print("GRU-AE TRAINING")
    print("=" * 60)

    print()

    print(
        "Device:",
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
                ).total_memory / 1024**3,
                2
            ),
            "GB"
        )

    print()

    X_train = np.load(
        os.path.join(
            DATA_DIR,
            "X_train.npy"
        )
    )

    X_val = np.load(
        os.path.join(
            DATA_DIR,
            "X_val.npy"
        )
    )

    print(
        "Training data:",
        X_train.shape
    )

    print(
        "Validation data:",
        X_val.shape
    )

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        )
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        pin_memory=True
    )

    model = GRUAutoencoder(
        input_size=9
    ).to(
        DEVICE
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        parameters
    )

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 300

    best_val = float(
        "inf"
    )

    patience = 30

    no_improvement = 0

    best_path = os.path.join(
        MODEL_DIR,
        "gru_ae_best.pt"
    )

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        train_total = 0.0

        for (x,) in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            reconstruction = model(
                x
            )

            loss = criterion(
                reconstruction,
                x
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_total += (
                loss.item() *
                x.size(0)
            )

        train_loss = (
            train_total /
            len(train_loader.dataset)
        )

        model.eval()

        val_total = 0.0

        with torch.no_grad():

            for (x,) in val_loader:

                x = x.to(
                    DEVICE,
                    non_blocking=True
                )

                reconstruction = model(
                    x
                )

                loss = criterion(
                    reconstruction,
                    x
                )

                val_total += (
                    loss.item() *
                    x.size(0)
                )

        val_loss = (
            val_total /
            len(val_loader.dataset)
        )

        history.append(
            [
                epoch,
                train_loss,
                val_loss
            ]
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "val_loss":
                        val_loss,
                    "input_size":
                        9
                },
                best_path
            )

            print(
                "  New best model saved."
            )

        else:

            no_improvement += 1

        if no_improvement >= patience:

            print()
            print(
                "Early stopping."
            )

            break

    np.save(
        os.path.join(
            RESULTS_DIR,
            "gru_training_history.npy"
        ),
        np.asarray(
            history
        )
    )

    print()
    print("=" * 60)
    print("GRU-AE TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best validation loss:",
        best_val
    )

    print(
        "Model:",
        best_path
    )


if __name__ == "__main__":

    main()