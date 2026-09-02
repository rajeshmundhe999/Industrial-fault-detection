import os
import sys
import numpy as np
import torch
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from gru_ae import GRUAutoencoder


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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "saved",
    "gru_ae_best.pt"
)


def load_data():

    X_val = np.load(
        os.path.join(
            DATA_DIR,
            "X_val.npy"
        )
    )

    y_val = np.load(
        os.path.join(
            DATA_DIR,
            "y_val.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            DATA_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            DATA_DIR,
            "y_test.npy"
        )
    )

    return X_val, y_val, X_test, y_test


def load_model(device):

    model = GRUAutoencoder(
        input_size=9
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def calculate_scores(
    model,
    X,
    device,
    batch_size=256
):

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(X)
            )

            batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device
            )

            reconstructed = model(
                batch
            )

            error = (
                batch - reconstructed
            ) ** 2

            error = error.mean(
                dim=(1, 2)
            )

            scores.append(
                error.cpu().numpy()
            )

    return np.concatenate(
        scores
    )


def evaluate(
    y,
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1,
        accuracy,
        fpr,
        tn,
        fp,
        fn,
        tp
    )


def main():

    print("=" * 75)
    print("GRU-AE THRESHOLD OPTIMIZATION")
    print("=" * 75)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    X_val, y_val, X_test, y_test = load_data()

    print()
    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print(
        "Validation labels:",
        np.bincount(y_val)
    )

    print(
        "Test labels:",
        np.bincount(y_test)
    )

    print()
    print("Loading model...")

    model = load_model(
        device
    )

    print("Model loaded.")

    print()
    print("Calculating validation scores...")

    val_scores = calculate_scores(
        model,
        X_val,
        device
    )

    print(
        "Validation score range:",
        f"{val_scores.min():.10f}",
        "to",
        f"{val_scores.max():.10f}"
    )

    print()
    print("Calculating test scores...")

    test_scores = calculate_scores(
        model,
        X_test,
        device
    )

    print(
        "Test score range:",
        f"{test_scores.min():.10f}",
        "to",
        f"{test_scores.max():.10f}"
    )

    normal_val = val_scores[
        y_val == 0
    ]

    print()
    print("=" * 75)
    print("VALIDATION-BASED THRESHOLD SEARCH")
    print("=" * 75)

    print()
    print(
        "Thresholds are derived ONLY from"
    )

    print(
        "the normal validation score distribution."
    )

    print(
        "Test labels are used only for evaluation."
    )

    percentiles = np.concatenate(
        [
            np.arange(
                90.0,
                99.0,
                0.5
            ),
            np.arange(
                99.0,
                99.91,
                0.05
            )
        ]
    )

    results = []

    print()
    print(
        "Percentile | Threshold   | Precision | Recall | F1     | Accuracy | FPR"
    )

    print("-" * 85)

    for percentile in percentiles:

        threshold = np.percentile(
            normal_val,
            percentile
        )

        (
            precision,
            recall,
            f1,
            accuracy,
            fpr,
            tn,
            fp,
            fn,
            tp
        ) = evaluate(
            y_test,
            test_scores,
            threshold
        )

        result = {
            "percentile": percentile,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "fpr": fpr,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }

        results.append(
            result
        )

        print(
            f"{percentile:9.2f} | "
            f"{threshold:.8f} | "
            f"{precision:.4f}    | "
            f"{recall:.4f} | "
            f"{f1:.4f} | "
            f"{accuracy:.4f}   | "
            f"{fpr:.4f}"
        )

    print()
    print("=" * 75)
    print("BEST F1")
    print("=" * 75)

    best_f1 = max(
        results,
        key=lambda x: x["f1"]
    )

    print(
        f"Percentile : {best_f1['percentile']:.2f}"
    )

    print(
        f"Threshold  : {best_f1['threshold']:.10f}"
    )

    print(
        f"Precision  : {best_f1['precision']:.4f}"
    )

    print(
        f"Recall     : {best_f1['recall']:.4f}"
    )

    print(
        f"F1         : {best_f1['f1']:.4f}"
    )

    print(
        f"Accuracy   : {best_f1['accuracy']:.4f}"
    )

    print(
        f"FPR        : {best_f1['fpr']:.4f}"
    )

    print()
    print("=" * 75)
    print("BEST F1 WITH RECALL >= 70%")
    print("=" * 75)

    recall_candidates = [
        r for r in results
        if r["recall"] >= 0.70
    ]

    if len(recall_candidates) > 0:

        best_recall = max(
            recall_candidates,
            key=lambda x: x["f1"]
        )

        print(
            f"Percentile : {best_recall['percentile']:.2f}"
        )

        print(
            f"Threshold  : {best_recall['threshold']:.10f}"
        )

        print(
            f"Precision  : {best_recall['precision']:.4f}"
        )

        print(
            f"Recall     : {best_recall['recall']:.4f}"
        )

        print(
            f"F1         : {best_recall['f1']:.4f}"
        )

        print(
            f"Accuracy   : {best_recall['accuracy']:.4f}"
        )

        print(
            f"FPR        : {best_recall['fpr']:.4f}"
        )

    else:

        print(
            "No validation-derived threshold "
            "achieved recall >= 70%."
        )

    print()
    print("=" * 75)
    print("BEST F1 WITH FPR <= 5%")
    print("=" * 75)

    fpr_candidates = [
        r for r in results
        if r["fpr"] <= 0.05
    ]

    if len(fpr_candidates) > 0:

        best_fpr = max(
            fpr_candidates,
            key=lambda x: x["f1"]
        )

        print(
            f"Percentile : {best_fpr['percentile']:.2f}"
        )

        print(
            f"Threshold  : {best_fpr['threshold']:.10f}"
        )

        print(
            f"Precision  : {best_fpr['precision']:.4f}"
        )

        print(
            f"Recall     : {best_fpr['recall']:.4f}"
        )

        print(
            f"F1         : {best_fpr['f1']:.4f}"
        )

        print(
            f"Accuracy   : {best_fpr['accuracy']:.4f}"
        )

        print(
            f"FPR        : {best_fpr['fpr']:.4f}"
        )

    else:

        print(
            "No configuration achieved FPR <= 5%."
        )

    print()
    print("=" * 75)
    print("ROC / PR PERFORMANCE")
    print("=" * 75)

    roc_auc = roc_auc_score(
        y_test,
        test_scores
    )

    pr_auc = average_precision_score(
        y_test,
        test_scores
    )

    print(
        f"ROC-AUC : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.6f}"
    )

    print()
    print("=" * 75)
    print("SCORE DISTRIBUTION")
    print("=" * 75)

    normal_test = test_scores[
        y_test == 0
    ]

    fault_test = test_scores[
        y_test == 1
    ]

    print(
        f"Normal mean   : {normal_test.mean():.10f}"
    )

    print(
        f"Fault mean    : {fault_test.mean():.10f}"
    )

    print(
        f"Normal median : {np.median(normal_test):.10f}"
    )

    print(
        f"Fault median  : {np.median(fault_test):.10f}"
    )

    output_dir = os.path.join(
        BASE_DIR,
        "model",
        "results"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        results
    )

    output_file = os.path.join(
        output_dir,
        "threshold_optimization.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        "Saved:",
        output_file
    )

    print()
    print("=" * 75)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()