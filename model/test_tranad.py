import os
import sys

import numpy as np
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.tranad import (
    TranAD,
    reconstruction_loss
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "X_train.npy"
)


def main():

    print("=" * 60)
    print("TRANAD GPU SMOKE TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    data = np.load(
        DATA_FILE,
        mmap_mode="r"
    )

    print()
    print("Training data shape:", data.shape)

    sample = torch.tensor(
        data[:32],
        dtype=torch.float32
    ).to(device)

    print(
        "Test batch shape:",
        tuple(sample.shape)
    )

    model = TranAD(
        n_features=9,
        window_size=60
    ).to(device)

    print()
    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    model.train()

    optimizer.zero_grad()

    reconstruction_1, reconstruction_2 = (
        model(sample)
    )

    loss_1 = reconstruction_loss(
        sample,
        reconstruction_1
    )

    loss_2 = reconstruction_loss(
        sample,
        reconstruction_2
    )

    loss = loss_1 + loss_2

    print()
    print(
        "Reconstruction 1 shape:",
        tuple(reconstruction_1.shape)
    )

    print(
        "Reconstruction 2 shape:",
        tuple(reconstruction_2.shape)
    )

    print(
        "Loss 1:",
        loss_1.item()
    )

    print(
        "Loss 2:",
        loss_2.item()
    )

    print(
        "Total loss:",
        loss.item()
    )

    loss.backward()

    optimizer.step()

    print()
    print("Backward pass: SUCCESS")
    print("Optimizer step: SUCCESS")

    if device.type == "cuda":

        print(
            "GPU memory allocated:",
            round(
                torch.cuda.memory_allocated()
                / 1024**3,
                3
            ),
            "GB"
        )

    print()
    print("=" * 60)
    print("TRANAD GPU SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()