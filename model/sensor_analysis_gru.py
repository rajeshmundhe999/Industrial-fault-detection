import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from gru_ae import GRUAutoencoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved")
RESULT_DIR = os.path.join(BASE_DIR, "model", "results")
PLOT_DIR = os.path.join(BASE_DIR, "model", "plots")

MODEL_PATH = os.path.join(MODEL_DIR, "gru_ae_best.pt")

TEST_DATA_PATH = os.path.join(DATA_DIR, "X_test.npy")
TEST_LABEL_PATH = os.path.join(RESULT_DIR, "test_labels.npy")

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

os.makedirs(PLOT_DIR, exist_ok=True)


def main():

    print("=" * 70)
    print("GRU-AE SENSOR-LEVEL ANOMALY ANALYSIS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nDevice:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    X_test = np.load(TEST_DATA_PATH)
    y_test = np.load(TEST_LABEL_PATH)

    print("\nTest data:", X_test.shape)
    print("Labels:", y_test.shape)

    model = GRUAutoencoder(
    input_size=9
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded.")

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    print("\nCalculating sensor reconstruction errors...")

    errors = []

    batch_size = 256

    with torch.no_grad():

        for i in range(0, len(X_tensor), batch_size):

            batch = X_tensor[i:i + batch_size]

            reconstructed = model(batch)

            error = torch.abs(
                batch - reconstructed
            )

            error = error.cpu().numpy()

            errors.append(error)

    errors = np.concatenate(errors, axis=0)

    print("Error shape:", errors.shape)

    sensor_errors = errors.mean(axis=1)

    normal_mask = y_test == 0
    fault_mask = y_test == 1

    normal_sensor_error = sensor_errors[normal_mask].mean(axis=0)
    fault_sensor_error = sensor_errors[fault_mask].mean(axis=0)

    contribution = (
        fault_sensor_error /
        (normal_sensor_error + 1e-12)
    )

    ranking = np.argsort(contribution)[::-1]

    print("\n" + "=" * 70)
    print("SENSOR CONTRIBUTION RANKING")
    print("=" * 70)

    print(
        "\n{:>4} {:<30} {:>15} {:>15} {:>15}".format(
            "Rank",
            "Sensor",
            "Normal Error",
            "Fault Error",
            "Ratio"
        )
    )

    print("-" * 85)

    for rank, idx in enumerate(ranking, 1):

        print(
            "{:>4} {:<30} {:>15.8f} {:>15.8f} {:>15.2f}x".format(
                rank,
                SENSOR_NAMES[idx],
                normal_sensor_error[idx],
                fault_sensor_error[idx],
                contribution[idx]
            )
        )

    np.save(
        os.path.join(
            RESULT_DIR,
            "gru_sensor_errors.npy"
        ),
        sensor_errors
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "gru_sensor_ranking.npy"
        ),
        ranking
    )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            "gru_sensor_contribution.csv"
        ),
        np.column_stack(
            (
                np.arange(1, 10),
                normal_sensor_error,
                fault_sensor_error,
                contribution
            )
        ),
        delimiter=",",
        header="sensor_index,normal_error,fault_error,fault_normal_ratio",
        comments=""
    )

    plt.figure(figsize=(12, 6))

    x = np.arange(len(SENSOR_NAMES))
    width = 0.38

    plt.bar(
        x - width / 2,
        normal_sensor_error,
        width,
        label="Normal"
    )

    plt.bar(
        x + width / 2,
        fault_sensor_error,
        width,
        label="Fault"
    )

    plt.xticks(
        x,
        SENSOR_NAMES,
        rotation=45,
        ha="right"
    )

    plt.ylabel("Mean Reconstruction Error")
    plt.title("GRU-AE Sensor Reconstruction Error")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        "gru_sensor_error_normal_vs_fault.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("\nSaved:", path)

    plt.figure(figsize=(12, 6))

    sorted_names = [
        SENSOR_NAMES[i]
        for i in ranking
    ]

    sorted_contribution = contribution[ranking]

    plt.bar(
        np.arange(len(sorted_names)),
        sorted_contribution
    )

    plt.xticks(
        np.arange(len(sorted_names)),
        sorted_names,
        rotation=45,
        ha="right"
    )

    plt.ylabel("Fault / Normal Reconstruction Error")
    plt.title("GRU-AE Sensor Contribution to Fault Detection")
    plt.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        "gru_sensor_contribution.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

    fault_errors = sensor_errors[fault_mask]

    plt.figure(figsize=(12, 7))

    plt.imshow(
        fault_errors.T,
        aspect="auto",
        interpolation="nearest"
    )

    plt.yticks(
        np.arange(len(SENSOR_NAMES)),
        SENSOR_NAMES
    )

    plt.xlabel("Fault Test Window")
    plt.ylabel("Sensor")
    plt.title("GRU-AE Fault Reconstruction Error Heatmap")

    plt.colorbar(
        label="Absolute Reconstruction Error"
    )

    plt.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        "gru_fault_sensor_heatmap.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)

    print("\n" + "=" * 70)
    print("TOP 3 CONTRIBUTING SENSORS")
    print("=" * 70)

    for i in ranking[:3]:

        print(
            SENSOR_NAMES[i],
            "->",
            round(contribution[i], 2),
            "x higher error during faults"
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()