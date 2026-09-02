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


def gru_scores(model, X, device):

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

            reconstructed = model(
                batch
            )

            error = (
                batch - reconstructed
            ) ** 2

            score = error.mean(
                dim=(1, 2)
            )

            scores.append(
                score.cpu().numpy()
            )

    return np.concatenate(scores)


def tranad_scores(model, X, device):

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

    return np.concatenate(scores)


def normalize(
    scores,
    low,
    high
):

    return (
        scores - low
    ) / (
        high - low + 1e-12
    )


def evaluate(
    name,
    scores,
    labels,
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

    auc = roc_auc_score(
        labels,
        scores
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (
        fp + tn
    )

    print(
        "{:<25} "
        "F1: {:.4f} | "
        "AUC: {:.4f} | "
        "Precision: {:.4f} | "
        "Recall: {:.4f} | "
        "FPR: {:.4f}".format(
            name,
            f1,
            auc,
            precision,
            recall,
            fpr
        )
    )

    return {
        "name": name,
        "f1": f1,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "fpr": fpr,
        "threshold": threshold
    }


def main():

    print("=" * 70)
    print("GRU + TRANAD HYBRID EVALUATION")
    print("=" * 70)

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
        "\nCalculating GRU validation scores..."
    )

    gru_val = gru_scores(
        gru,
        X_VAL,
        device
    )

    print(
        "Calculating GRU test scores..."
    )

    gru_test = gru_scores(
        gru,
        X_TEST,
        device
    )

    print(
        "Calculating TranAD validation scores..."
    )

    tranad_val = tranad_scores(
        tranad,
        X_VAL,
        device
    )

    print(
        "Calculating TranAD test scores..."
    )

    tranad_test = tranad_scores(
        tranad,
        X_TEST,
        device
    )

    gru_low = np.percentile(
        gru_val,
        1
    )

    gru_high = np.percentile(
        gru_val,
        99
    )

    tranad_low = np.percentile(
        tranad_val,
        1
    )

    tranad_high = np.percentile(
        tranad_val,
        99
    )

    gru_val_n = normalize(
        gru_val,
        gru_low,
        gru_high
    )

    gru_test_n = normalize(
        gru_test,
        gru_low,
        gru_high
    )

    tranad_val_n = normalize(
        tranad_val,
        tranad_low,
        tranad_high
    )

    tranad_test_n = normalize(
        tranad_test,
        tranad_low,
        tranad_high
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TESTING FUSION STRATEGIES"
    )

    print(
        "=" * 70
    )

    results = []

    configs = [
        (
            "GRU 100%",
            1.0,
            0.0
        ),
        (
            "TranAD 100%",
            0.0,
            1.0
        ),
        (
            "Fusion 50/50",
            0.5,
            0.5
        ),
        (
            "Fusion 60/40",
            0.6,
            0.4
        ),
        (
            "Fusion 70/30",
            0.7,
            0.3
        ),
        (
            "Fusion 80/20",
            0.8,
            0.2
        ),
        (
            "Fusion 30/70",
            0.3,
            0.7
        ),
        (
            "Fusion 40/60",
            0.4,
            0.6
        )
    ]

    for name, gw, tw in configs:

        val_fusion = (
            gw * gru_val_n
            +
            tw * tranad_val_n
        )

        test_fusion = (
            gw * gru_test_n
            +
            tw * tranad_test_n
        )

        threshold = np.percentile(
            val_fusion,
            99.9
        )

        result = evaluate(
            name,
            test_fusion,
            Y_TEST,
            threshold
        )

        results.append(
            result
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST FUSION RESULT"
    )

    print(
        "=" * 70
    )

    best = max(
        results,
        key=lambda x: x["f1"]
    )

    print(
        "\nModel:",
        best["name"]
    )

    print(
        "F1:",
        best["f1"]
    )

    print(
        "ROC-AUC:",
        best["auc"]
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
        "Accuracy:",
        best["accuracy"]
    )

    print(
        "FPR:",
        best["fpr"]
    )

    print(
        "Threshold:",
        best["threshold"]
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "gru_test_scores_hybrid.npy"
        ),
        gru_test_n
    )

    np.save(
        os.path.join(
            RESULT_DIR,
            "tranad_test_scores_hybrid.npy"
        ),
        tranad_test_n
    )

    print(
        "\nResults saved."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "HYBRID EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()