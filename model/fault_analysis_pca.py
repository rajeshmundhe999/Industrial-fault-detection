import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)


def main():

    print("=" * 70)
    print("PCA / MSPC FAULT-WISE ANALYSIS")
    print("=" * 70)

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

    spe = np.load(
        os.path.join(
            RESULT_DIR,
            "pca_test_spe.npy"
        )
    )

    t2 = np.load(
        os.path.join(
            RESULT_DIR,
            "pca_test_t2.npy"
        )
    )

    val_spe = np.load(
        os.path.join(
            RESULT_DIR,
            "pca_val_spe.npy"
        )
    )

    val_t2 = np.load(
        os.path.join(
            RESULT_DIR,
            "pca_val_t2.npy"
        )
    )

    spe_threshold = np.percentile(
        val_spe,
        98.0
    )

    t2_threshold = np.percentile(
        val_t2,
        99.7
    )

    predictions = (
        (spe >= spe_threshold)
        |
        (t2 >= t2_threshold)
    ).astype(int)

    print(
        "\nThresholds:"
    )

    print(
        "SPE:",
        spe_threshold
    )

    print(
        "T2:",
        t2_threshold
    )

    print(
        "\nOverall:"
    )

    print(
        "Precision:",
        precision_score(
            labels,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            labels,
            predictions,
            zero_division=0
        )
    )

    print(
        "F1:",
        f1_score(
            labels,
            predictions,
            zero_division=0
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FAULT-WISE PERFORMANCE"
    )

    print(
        "=" * 70
    )

    rows = []

    for batch in sorted(
        np.unique(batch_ids)
    ):

        mask = (
            batch_ids == batch
        )

        fault_mask = (
            mask
            &
            (labels == 1)
        )

        if not np.any(fault_mask):
            continue

        y_true = labels[mask]
        y_pred = predictions[mask]

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        fault_detected = np.sum(
            predictions[fault_mask]
        )

        fault_total = np.sum(
            fault_mask
        )

        normal_mask = (
            mask
            &
            (labels == 0)
        )

        normal_mean_spe = (
            np.mean(
                spe[normal_mask]
            )
            if np.any(normal_mask)
            else 0
        )

        fault_mean_spe = np.mean(
            spe[fault_mask]
        )

        normal_mean_t2 = (
            np.mean(
                t2[normal_mask]
            )
            if np.any(normal_mask)
            else 0
        )

        fault_mean_t2 = np.mean(
            t2[fault_mask]
        )

        rows.append(
            {
                "batch": int(batch),
                "fault_windows": int(
                    fault_total
                ),
                "detected": int(
                    fault_detected
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "normal_spe": normal_mean_spe,
                "fault_spe": fault_mean_spe,
                "normal_t2": normal_mean_t2,
                "fault_t2": fault_mean_t2
            }
        )

        print(
            "{:5d} | {:6d} | {:8d} | "
            "{:.4f} | {:.4f} | {:.4f} | "
            "SPE {:.6f} → {:.6f} | "
            "T2 {:.2f} → {:.2f}".format(
                int(batch),
                int(fault_total),
                int(fault_detected),
                precision,
                recall,
                f1,
                normal_mean_spe,
                fault_mean_spe,
                normal_mean_t2,
                fault_mean_t2
            )
        )

    output = pd.DataFrame(
        rows
    )

    output_path = os.path.join(
        RESULT_DIR,
        "pca_fault_wise_performance.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved:",
        output_path
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PCA FAULT-WISE ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()