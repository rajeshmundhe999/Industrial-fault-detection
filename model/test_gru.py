import torch
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.gru_ae import GRUAutoencoder


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("GRU-AE GPU SMOKE TEST")
print("=" * 60)

print()

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

model = GRUAutoencoder(
    input_size=9
).to(device)

x = torch.randn(
    32,
    60,
    9
).to(device)

y = model(x)

loss = torch.mean(
    (y - x) ** 2
)

print()

print(
    "Input:",
    tuple(x.shape)
)

print(
    "Output:",
    tuple(y.shape)
)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print(
    "Loss:",
    loss.item()
)

loss.backward()

print()

print(
    "Backward pass: SUCCESS"
)

print("=" * 60)
print("GRU-AE SMOKE TEST PASSED")
print("=" * 60)