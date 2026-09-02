import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

from gru_ae import GRUAutoencoder


BASE_DIR = r"C:\000 VIT\ETHANOL\pfdi_project"

X_FILE = BASE_DIR + r"\data\processed\X_test.npy"
FAULT_FILE = BASE_DIR + r"\data\processed\batch_faults.npy"
BATCH_FILE = BASE_DIR + r"\data\processed\batch_ids_test.npy"

MODEL_FILE = BASE_DIR + r"\model\saved\gru_ae_best.pt"

WINDOW_SIZE = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("LABEL SENSITIVITY ANALYSIS")
print("=" * 70)

X = np.load(X_FILE)
batch_ids = np.load(BATCH_FILE)

faults = np.load(
    FAULT_FILE,
    allow_pickle=True
).item()

print()
print("Test shape:", X.shape)
print("Device:", device)

window_purity = []

for i in range(len(X)):

    batch = str(batch_ids[i])

    fault = faults[batch]["fault"]

    n_rows = len(fault)

    batch_windows = (n_rows - WINDOW_SIZE) // 5 + 1

    local_index = np.sum(
        batch_ids[:i] == batch_ids[i]
    )

    start = local_index * 5
    end = start + WINDOW_SIZE

    if end > n_rows:
        end = n_rows

    purity = np.mean(
        fault[start:end] > 0
    )

    window_purity.append(purity)

window_purity = np.asarray(
    window_purity,
    dtype=np.float32
)

print()
print("=" * 70)
print("WINDOW PURITY")
print("=" * 70)

print("Minimum :", window_purity.min())
print("Mean    :", window_purity.mean())
print("Median  :", np.median(window_purity))
print("Maximum :", window_purity.max())

print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = GRUAutoencoder(
    input_size=9
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded.")

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

scores = []

with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        256
    ):

        batch = X_tensor[
            start:start + 256
        ].to(device)

        output = model(batch)

        if isinstance(output, tuple):
            reconstruction = output[0]
        else:
            reconstruction = output

        error = (
            (batch - reconstruction) ** 2
        ).mean(
            dim=(1, 2)
        )

        scores.extend(
            error.cpu().numpy()
        )

scores = np.asarray(scores)

print()
print("=" * 70)
print("SCORE PERFORMANCE")
print("=" * 70)

for purity_threshold in [
    0.0,
    0.10,
    0.25,
    0.50,
    0.75
]:

    if purity_threshold == 0:
        labels = (
            window_purity > 0
        ).astype(int)

        name = "ANY"

    else:
        labels = (
            window_purity >= purity_threshold
        ).astype(int)

        name = f">={purity_threshold:.2f}"

    auc = roc_auc_score(
        labels,
        scores
    )

    pr_auc = average_precision_score(
        labels,
        scores
    )

    normal_scores = scores[
        labels == 0
    ]

    threshold = np.percentile(
        normal_scores,
        98
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    print()
    print(
        f"LABEL: {name}"
    )

    print(
        f"Fault windows: {labels.sum()}"
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"PR-AUC : {pr_auc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print("Confusion matrix:")
    print(cm)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)