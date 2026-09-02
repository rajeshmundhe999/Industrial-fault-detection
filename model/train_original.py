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

from model.tranad_original import TranADOriginal


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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 60)
    print("REFERENCE TRANAD TRAINING")
    print("=" * 60)
    print()

    print("Device:", DEVICE)

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
        "Original training data:",
        X_train.shape
    )

    print(
        "Original validation data:",
        X_val.shape
    )

    window = 10

    X_train = X_train[
        :,
        -window:,
        :
    ]

    X_val = X_val[
        :,
        -window:,
        :
    ]

    print(
        "TranAD training data:",
        X_train.shape
    )

    print(
        "TranAD validation data:",
        X_val.shape
    )

    print()

    train_source = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    train_target = torch.tensor(
        X_train[:, -1, :],
        dtype=torch.float32
    )

    val_source = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    val_target = torch.tensor(
        X_val[:, -1, :],
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        train_source,
        train_target
    )

    val_dataset = TensorDataset(
        val_source,
        val_target
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True
    )

    model = TranADOriginal(
        X_train.shape[2]
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

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.9
    )

    epochs = 50

    best_val = float("inf")

    patience = 8

    no_improvement = 0

    best_path = os.path.join(
        MODEL_DIR,
        "tranad_original_best.pt"
    )

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        train_total = 0.0

        for src, target in train_loader:

            src = src.to(
                DEVICE,
                non_blocking=True
            )

            target = target.to(
                DEVICE,
                non_blocking=True
            )

            src = src.transpose(
                0,
                1
            )

            target = target.unsqueeze(
                0
            )

            x1, x2 = model(
                src,
                target
            )

            loss1 = criterion(
                x1,
                target
            )

            loss2 = criterion(
                x2,
                target
            )

            loss = (
                0.5 * loss1 +
                0.5 * loss2
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
                src.size(1)
            )

        train_loss = (
            train_total /
            len(train_loader.dataset)
        )

        model.eval()

        val_total = 0.0

        with torch.no_grad():

            for src, target in val_loader:

                src = src.to(
                    DEVICE,
                    non_blocking=True
                )

                target = target.to(
                    DEVICE,
                    non_blocking=True
                )

                src = src.transpose(
                    0,
                    1
                )

                target = target.unsqueeze(
                    0
                )

                x1, x2 = model(
                    src,
                    target
                )

                loss1 = criterion(
                    x1,
                    target
                )

                loss2 = criterion(
                    x2,
                    target
                )

                loss = (
                    0.5 * loss1 +
                    0.5 * loss2
                )

                val_total += (
                    loss.item() *
                    src.size(1)
                )

        val_loss = (
            val_total /
            len(val_loader.dataset)
        )

        scheduler.step()

        lr = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {lr:.2e}"
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
                    "window":
                        window,
                    "n_features":
                        X_train.shape[2]
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

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
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