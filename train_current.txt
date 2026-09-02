import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tranad import TranAD


SEED = 42

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4

PATIENCE = 7

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


def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


def load_data():

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

    return X_train, X_val


def create_loader(data, shuffle):

    tensor = torch.tensor(
        data,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return loader


def calculate_loss(
    model,
    batch,
    device
):

    batch = batch.to(
        device,
        non_blocking=True
    )

    reconstruction_1, reconstruction_2 = (
        model(batch)
    )

    loss_1 = torch.mean(
        torch.abs(
            batch - reconstruction_1
        )
    )

    loss_2 = torch.mean(
        torch.abs(
            batch - reconstruction_2
        )
    )

    loss = (
        0.5 * loss_1 +
        0.5 * loss_2
    )

    return loss


def train_epoch(
    model,
    loader,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    for (batch,) in tqdm(
        loader,
        desc="Training",
        leave=False
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = calculate_loss(
            model,
            batch,
            device
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item() *
            batch.size(0)
        )

    return (
        total_loss /
        len(loader.dataset)
    )


def validate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for (batch,) in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):

            loss = calculate_loss(
                model,
                batch,
                device
            )

            total_loss += (
                loss.item() *
                batch.size(0)
            )

    return (
        total_loss /
        len(loader.dataset)
    )


def main():

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("TRANAD TRAINING")
    print("=" * 60)

    print()

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

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

    X_train, X_val = load_data()

    print()

    print(
        "Training data:",
        X_train.shape
    )

    print(
        "Validation data:",
        X_val.shape
    )

    train_loader = create_loader(
        X_train,
        shuffle=True
    )

    val_loader = create_loader(
        X_val,
        shuffle=False
    )

    model = TranAD(
        n_features=9,
        window_size=60
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    train_history = []

    val_history = []

    print()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            device
        )

        scheduler.step(
            val_loss
        )

        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )

        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Val Loss:   {val_loss:.6f}"
        )

        print(
            f"LR:         {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "val_loss":
                        val_loss,

                    "n_features":
                        9,

                    "window_size":
                        60
                },
                os.path.join(
                    MODEL_DIR,
                    "tranad_best.pt"
                )
            )

            print(
                "✓ New best model saved."
            )

        else:

            epochs_without_improvement += 1

        print()

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "Early stopping."
            )

            break

    history = {

        "train_loss":
            np.array(
                train_history
            ),

        "val_loss":
            np.array(
                val_history
            )
    }

    np.save(
        os.path.join(
            RESULTS_DIR,
            "training_history.npy"
        ),
        history,
        allow_pickle=True
    )

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print()

    print(
        "Best validation loss:",
        best_val_loss
    )

    print(
        "Model:",
        os.path.join(
            MODEL_DIR,
            "tranad_best.pt"
        )
    )


if __name__ == "__main__":

    main()