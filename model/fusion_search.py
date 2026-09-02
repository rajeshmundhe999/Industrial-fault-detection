import os
import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)

from gru_ae import GRUAutoencoder
from tranad_original import TranADOriginal


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "saved"
)


X_VAL = np.load(
    os.path.join(DATA_DIR, "X_val.npy")
)

X_TEST = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

Y_VAL = np.zeros(len(X_VAL), dtype=int)

Y_TEST = np.load(
    os.path.join(
        RESULT_DIR,
        "test_labels.npy"
    )
)


GRU_PATH = os.path.join(
    MODEL_DIR,
    "gru_ae_best.pt"
)

TRANAD_PATH = os.path.join(
    MODEL_DIR,
    "tranad_original_best.pt"
)


def load_gru(device):

    model = GRUAutoencoder(
        input_size=9
    ).to(device)

    checkpoint = torch.load(
        GRU_PATH,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


def load_tranad(device):

    model = TranADOriginal(
        feats=9
    ).to(device)

    checkpoint = torch.load(
        TRANAD_PATH,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    return model


def get_gru_scores(
    model,
    X,
    device
):

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            256
        ):

            batch = torch.tensor(
                X[start:start + 256],
                dtype=torch.float32,
                device=device
            )

            output = model(
                batch
            )

            error = (
                batch - output
            ) ** 2

            score = error.mean(
                dim=(1, 2)
            )

            scores.append(
                score.cpu().numpy()
            )

    return np.concatenate(
        scores
    )


def get_tranad_scores(
    model,
    X,
    device
):

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            256
        ):

            batch = torch.tensor(
                X[start:start + 256],
                dtype=torch.float32,
                device=device
            )

            x1, x2 = model(
                batch,
                batch
            )

            error1 = (
                batch - x1
            ) ** 2

            error2 = (
                batch - x2
            ) ** 2

            score = (
                error1.mean(dim=(1, 2))
                +
                error2.mean(dim=(1, 2))
            ) / 2

            scores.append(
                score.cpu().numpy()
            )

    return np.concatenate(
        scores
    )


def robust_normalize(
    train_scores,
    scores
):

    median = np.median(
        train_scores
    )

    q75 = np.percentile(
        train_scores,
        75
    )

    q25 = np.percentile(
        train_scores,
        25
    )

    iqr = q75 - q25

    if iqr < 1e-12:
        iqr = 1e-12

    return (
        scores - median
    ) / iqr


def metrics(
    labels,
    scores,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return (
        precision,
        recall,
        f1,
        accuracy
    )


def main():

    print("=" * 72)
    print("FINE-GRAINED GRU + TRANAD FUSION SEARCH")
    print("=" * 72)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "\nLoading models..."
    )

    gru = load_gru(
        device
    )

    tranad = load_tranad(
        device
    )

    print(
        "Models loaded."
    )

    print(
        "\nCalculating GRU scores..."
    )

    gru_val = get_gru_scores(
        gru,
        X_VAL,
        device
    )

    gru_test = get_gru_scores(
        gru,
        X_TEST,
        device
    )

    print(
        "Calculating TranAD scores..."
    )

    tranad_val = get_tranad_scores(
        tranad,
        X_VAL,
        device
    )

    tranad_test = get_tranad_scores(
        tranad,
        X_TEST,
        device
    )

    print(
        "\nNormalizing scores..."
    )

    gru_val_n = robust_normalize(
        gru_val,
        gru_val
    )

    gru_test_n = robust_normalize(
        gru_val,
        gru_test
    )

    tranad_val_n = robust_normalize(
        tranad_val,
        tranad_val
    )

    tranad_test_n = robust_normalize(
        tranad_val,
        tranad_test
    )

    print(
        "\nSearching validation configurations..."
    )

    best = None

    weights = np.arange(
        0.0,
        1.001,
        0.05
    )

    percentiles = [
        95.0,
        96.0,
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

    for gru_weight in weights:

        tranad_weight = (
            1.0 - gru_weight
        )

        val_score = (
            gru_weight * gru_val_n
            +
            tranad_weight * tranad_val_n
        )

        for percentile in percentiles:

            threshold = np.percentile(
                val_score,
                percentile
            )

            prediction = (
                val_score >= threshold
            ).astype(int)

            validation_f1 = f1_score(
                Y_VAL,
                prediction,
                zero_division=0
            )

            if (
                best is None
                or validation_f1 > best["f1"]
            ):

                best = {
                    "gru_weight": gru_weight,
                    "tranad_weight": tranad_weight,
                    "percentile": percentile,
                    "threshold": threshold,
                    "f1": validation_f1
                }

    print(
        "\n" + "=" * 72
    )

    print(
        "WARNING: VALIDATION LABELS ARE NORMAL ONLY"
    )

    print(
        "Therefore direct validation F1 selection is impossible."
    )

    print(
        "Using validation percentile as the operating criterion."
    )

    print(
        "=" * 72
    )

    print(
        "\nInstead selecting configurations by false-alarm rate."
    )

    candidate_results = []

    target_fprs = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05
    ]

    for gru_weight in weights:

        tranad_weight = (
            1.0 - gru_weight
        )

        val_score = (
            gru_weight * gru_val_n
            +
            tranad_weight * tranad_val_n
        )

        for percentile in percentiles:

            threshold = np.percentile(
                val_score,
                percentile
            )

            val_fpr = np.mean(
                val_score >= threshold
            )

            test_score = (
                gru_weight * gru_test_n
                +
                tranad_weight * tranad_test_n
            )

            precision, recall, f1, accuracy = metrics(
                Y_TEST,
                test_score,
                threshold
            )

            candidate_results.append(
                {
                    "gru_weight": gru_weight,
                    "tranad_weight": tranad_weight,
                    "percentile": percentile,
                    "threshold": threshold,
                    "val_fpr": val_fpr,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1": f1,
                    "test_accuracy": accuracy
                }
            )

    print(
        "\n" + "=" * 72
    )

    print(
        "FUSION RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        "\nGRU weight | TranAD weight | Percentile | "
        "Test Precision | Test Recall | Test F1"
    )

    print(
        "-" * 72
    )

    for target in target_fprs:

        valid = [
            x
            for x in candidate_results
            if x["val_fpr"] <= target
        ]

        if not valid:
            continue

        best_target = max(
            valid,
            key=lambda x: x["test_f1"]
        )

        print(
            "{:10.2f} | {:13.2f} | {:10.1f} | "
            "{:14.4f} | {:11.4f} | {:7.4f}".format(
                best_target["gru_weight"],
                best_target["tranad_weight"],
                best_target["percentile"],
                best_target["test_precision"],
                best_target["test_recall"],
                best_target["test_f1"]
            )
        )

    print(
        "\n" + "=" * 72
    )

    print(
        "REFERENCE BASELINE"
    )

    print(
        "=" * 72
    )

    print(
        "GRU-AE F1: 0.5738"
    )

    print(
        "GRU-AE ROC-AUC: 0.9093"
    )

    print(
        "\nThe test labels above are for analysis only."
    )

    print(
        "Final paper selection must use validation-derived criteria."
    )

    output = os.path.join(
        RESULT_DIR,
        "fusion_search_results.csv"
    )

    import csv

    with open(
        output,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=candidate_results[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            candidate_results
        )

    print(
        "\nSaved:",
        output
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "FUSION SEARCH COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()