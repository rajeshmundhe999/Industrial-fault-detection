import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)

SCORES = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_scores.npy"
    )
)

LABELS = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_labels.npy"
    )
)

BATCH_IDS = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_batch_ids.npy"
    )
)

TIMES = np.load(
    os.path.join(
        RESULTS_DIR,
        "test_times.npy"
    )
)

THRESHOLD = float(
    np.load(
        os.path.join(
            RESULTS_DIR,
            "threshold.npy"
        )
    )
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "model",
    "plots"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


def plot_score_distribution():

    normal_scores = SCORES[
        LABELS == 0
    ]

    fault_scores = SCORES[
        LABELS == 1
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        normal_scores,
        bins=100,
        alpha=0.6,
        label="Normal"
    )

    plt.hist(
        fault_scores,
        bins=100,
        alpha=0.6,
        label="Fault"
    )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        label="Threshold"
    )

    plt.xlabel(
        "Anomaly Score"
    )

    plt.ylabel(
        "Number of Windows"
    )

    plt.title(
        "TranAD Anomaly Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    output = os.path.join(
        PLOT_DIR,
        "score_distribution.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def plot_all_test_scores():

    plt.figure(
        figsize=(14, 6)
    )

    normal_mask = LABELS == 0
    fault_mask = LABELS == 1

    plt.plot(
        np.arange(len(SCORES)),
        SCORES,
        linewidth=0.8,
        alpha=0.7,
        label="Anomaly score"
    )

    plt.axhline(
        THRESHOLD,
        linestyle="--",
        label="Threshold"
    )

    fault_indices = np.where(
        fault_mask
    )[0]

    plt.scatter(
        fault_indices,
        SCORES[fault_mask],
        s=8,
        label="Fault windows"
    )

    plt.xlabel(
        "Test Window Index"
    )

    plt.ylabel(
        "Anomaly Score"
    )

    plt.title(
        "TranAD Anomaly Scores Across Test Set"
    )

    plt.legend()

    plt.tight_layout()

    output = os.path.join(
        PLOT_DIR,
        "test_anomaly_scores.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        "Saved:",
        output
    )


def plot_fault_batches():

    fault_batches = sorted(
        np.unique(
            BATCH_IDS[LABELS == 1]
        )
    )

    for batch in fault_batches:

        mask = (
            BATCH_IDS == batch
        )

        order = np.argsort(
            TIMES[mask]
        )

        times = TIMES[mask][order]
        scores = SCORES[mask][order]
        labels = LABELS[mask][order]

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            times,
            scores,
            linewidth=1.2,
            label="Anomaly score"
        )

        plt.axhline(
            THRESHOLD,
            linestyle="--",
            label="Threshold"
        )

        fault_mask = labels == 1

        if np.any(fault_mask):

            plt.scatter(
                times[fault_mask],
                scores[fault_mask],
                s=12,
                label="Fault"
            )

        plt.xlabel(
            "Time (h)"
        )

        plt.ylabel(
            "Anomaly Score"
        )

        plt.title(
            f"Batch {int(batch)} — TranAD Anomaly Score"
        )

        plt.legend()

        plt.tight_layout()

        output = os.path.join(
            PLOT_DIR,
            f"batch_{int(batch)}_anomaly_score.png"
        )

        plt.savefig(
            output,
            dpi=200
        )

        plt.close()

        print(
            "Saved:",
            output
        )


def main():

    print("=" * 60)
    print("ANOMALY SCORE ANALYSIS")
    print("=" * 60)

    print()

    print(
        "Threshold:",
        THRESHOLD
    )

    print(
        "Normal windows:",
        np.sum(LABELS == 0)
    )

    print(
        "Fault windows:",
        np.sum(LABELS == 1)
    )

    print()

    plot_score_distribution()

    plot_all_test_scores()

    plot_fault_batches()

    print()

    print("Analysis complete.")


if __name__ == "__main__":
    main()