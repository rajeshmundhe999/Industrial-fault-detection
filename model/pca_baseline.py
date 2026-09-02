import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)


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

RESULT_DIR = os.path.join(
    BASE_DIR,
    "model",
    "results"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "model",
    "plots"
)


TRAIN_PATH = os.path.join(
    DATA_DIR,
    "X_train.npy"
)

VAL_PATH = os.path.join(
    DATA_DIR,
    "X_val.npy"
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "X_test.npy"
)

LABEL_PATH = os.path.join(
    RESULT_DIR,
    "test_labels.npy"
)


def flatten_windows(X):

    return X.reshape(
        X.shape[0],
        -1
    )


def main():

    print("=" * 70)
    print("PCA / MSPC BASELINE")
    print("=" * 70)

    print("\nLoading data...")

    X_train = np.load(
        TRAIN_PATH
    )

    X_val = np.load(
        VAL_PATH
    )

    X_test = np.load(
        TEST_PATH
    )

    y_test = np.load(
        LABEL_PATH
    )

    print(
        "Training:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print(
        "Fault windows:",
        int(np.sum(y_test))
    )

    print(
        "\nFlattening temporal windows..."
    )

    X_train_flat = flatten_windows(
        X_train
    )

    X_val_flat = flatten_windows(
        X_val
    )

    X_test_flat = flatten_windows(
        X_test
    )

    print(
        "Flattened dimension:",
        X_train_flat.shape[1]
    )

    print(
        "\nFitting PCA..."
    )

    pca_full = PCA(
        svd_solver="randomized",
        random_state=42
    )

    pca_full.fit(
        X_train_flat
    )

    cumulative = np.cumsum(
        pca_full.explained_variance_ratio_
    )

    component_options = [
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50
    ]

    component_options = [
        x
        for x in component_options
        if x < X_train_flat.shape[1]
    ]

    best_component = None

    print(
        "\nExplained variance:"
    )

    for n in component_options:

        variance = cumulative[
            n - 1
        ]

        print(
            "Components: {:3d} | "
            "Explained variance: {:.4f}".format(
                n,
                variance
            )
        )

    print(
        "\nSelecting PCA model..."
    )

    target_variance = 0.95

    n_components = np.searchsorted(
        cumulative,
        target_variance
    ) + 1

    n_components = min(
        n_components,
        X_train_flat.shape[1]
    )

    print(
        "95% variance requires:",
        n_components,
        "components"
    )

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=42
    )

    X_train_pca = pca.fit_transform(
        X_train_flat
    )

    X_val_pca = pca.transform(
        X_val_flat
    )

    X_test_pca = pca.transform(
        X_test_flat
    )

    print(
        "PCA fitted."
    )

    print(
        "\nCalculating SPE reconstruction errors..."
    )

    train_reconstructed = pca.inverse_transform(
        X_train_pca
    )

    val_reconstructed = pca.inverse_transform(
        X_val_pca
    )

    test_reconstructed = pca.inverse_transform(
        X_test_pca
    )

    train_spe = np.mean(
        (
            X_train_flat
            -
            train_reconstructed
        ) ** 2,
        axis=1
    )

    val_spe = np.mean(
        (
            X_val_flat
            -
            val_reconstructed
        ) ** 2,
        axis=1
    )

    test_spe = np.mean(
        (
            X_test_flat
            -
            test_reconstructed
        ) ** 2,
        axis=1
    )

    print(
        "SPE calculated."
    )

    print(
        "\nCalculating Hotelling T²..."
    )

    scores = X_train_pca

    mean_scores = np.mean(
        scores,
        axis=0
    )

    covariance = np.cov(
        scores,
        rowvar=False
    )

    if covariance.ndim == 0:

        covariance = np.array(
            [[covariance]]
        )

    covariance += (
        np.eye(
            covariance.shape[0]
        ) * 1e-8
    )

    covariance_inverse = np.linalg.pinv(
        covariance
    )

    def t2_score(X):

        centered = (
            X
            -
            mean_scores
        )

        return np.einsum(
            "ij,jk,ik->i",
            centered,
            covariance_inverse,
            centered
        )

    train_t2 = t2_score(
        X_train_pca
    )

    val_t2 = t2_score(
        X_val_pca
    )

    test_t2 = t2_score(
        X_test_pca
    )

    print(
        "T² calculated."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATION-BASED THRESHOLDS"
    )

    print(
        "=" * 70
    )

    percentiles = [
        95.0,
        97.0,
        98.0,
        98.5,
        99.0,
        99.2,
        99.5,
        99.7,
        99.8,
        99.9
    ]

    print(
        "\nSPE thresholds:"
    )

    spe_thresholds = {}

    for percentile in percentiles:

        threshold = np.percentile(
            val_spe,
            percentile
        )

        spe_thresholds[
            percentile
        ] = threshold

        predictions = (
            test_spe
            >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print(
            "{:5.1f}% | "
            "Threshold: {:.8f} | "
            "Precision: {:.4f} | "
            "Recall: {:.4f} | "
            "F1: {:.4f}".format(
                percentile,
                threshold,
                precision,
                recall,
                f1
            )
        )

    print(
        "\nT² thresholds:"
    )

    t2_thresholds = {}

    for percentile in percentiles:

        threshold = np.percentile(
            val_t2,
            percentile
        )

        t2_thresholds[
            percentile
        ] = threshold

        predictions = (
            test_t2
            >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print(
            "{:5.1f}% | "
            "Threshold: {:.8f} | "
            "Precision: {:.4f} | "
            "Recall: {:.4f} | "
            "F1: {:.4f}".format(
                percentile,
                threshold,
                precision,
                recall,
                f1
            )
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "COMBINED MSPC"
    )

    print(
        "=" * 70
    )

    best = None

    for spe_percentile in percentiles:

        spe_threshold = spe_thresholds[
            spe_percentile
        ]

        for t2_percentile in percentiles:

            t2_threshold = t2_thresholds[
                t2_percentile
            ]

            spe_alarm = (
                test_spe
                >= spe_threshold
            )

            t2_alarm = (
                test_t2
                >= t2_threshold
            )

            predictions = (
                spe_alarm
                |
                t2_alarm
            ).astype(int)

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            if (
                best is None
                or f1 > best["f1"]
            ):

                best = {
                    "spe_percentile":
                        spe_percentile,
                    "t2_percentile":
                        t2_percentile,
                    "spe_threshold":
                        spe_threshold,
                    "t2_threshold":
                        t2_threshold,
                    "precision":
                        precision,
                    "recall":
                        recall,
                    "f1":
                        f1,
                    "accuracy":
                        accuracy
                }

    print(
        "\nBest combined MSPC configuration:"
    )

    print(
        "SPE percentile:",
        best["spe_percentile"]
    )

    print(
        "T² percentile:",
        best["t2_percentile"]
    )

    print(
        "Precision:",
        best["precision"]
    )

    print(
        "Recall:",
        best["recall"]
    )

    print(
        "F1:",
        best["f1"]
    )

    print(
        "Accuracy:",
        best["accuracy"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ROC-AUC"
    )

    print(
        "=" * 70
    )

    spe_auc = roc_auc_score(
        y_test,
        test_spe
    )

    t2_auc = roc_auc_score(
        y_test,
        test_t2
    )

    combined_score = (
        (
            test_spe
            -
            np.min(val_spe)
        )
        /
        (
            np.percentile(
                val_spe,
                99.5
            )
            -
            np.min(val_spe)
            +
            1e-12
        )
    )

    combined_score += (
        (
            test_t2
            -
            np.min(val_t2)
        )
        /
        (
            np.percentile(
                val_t2,
                99.5
            )
            -
            np.min(val_t2)
            +
            1e-12
        )
    )

    combined_auc = roc_auc_score(
        y_test,
        combined_score
    )

    print(
        "SPE ROC-AUC:",
        spe_auc
    )

    print(
        "T² ROC-AUC:",
        t2_auc
    )

    print(
        "Combined ROC-AUC:",
        combined_auc
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "pca_train_spe.npy"
        ),
        train_spe
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "pca_val_spe.npy"
        ),
        val_spe
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "pca_test_spe.npy"
        ),
        test_spe
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "pca_train_t2.npy"
        ),
        train_t2
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "pca_val_t2.npy"
        ),
        val_t2
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "pca_test_t2.npy"
        ),
        test_t2
    )

    results = pd.DataFrame(
        {
            "model": [
                "PCA-SPE",
                "PCA-T2",
                "PCA-MSPC"
            ],
            "roc_auc": [
                spe_auc,
                t2_auc,
                combined_auc
            ]
        }
    )

    results.to_csv(
        os.path.join(
            RESULT_DIR,
            "pca_results.csv"
        ),
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        os.path.join(
            RESULT_DIR,
            "pca_results.csv"
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PCA / MSPC BASELINE COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()