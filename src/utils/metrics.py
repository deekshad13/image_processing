import numpy as np


def verification_accuracy(similarities, labels, threshold=60):
    """
    Compute verification metrics at a given threshold.
    similarities: list of float (similarity percentages)
    labels: list of bool (True = genuine match, False = impostor)
    """
    if not similarities:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    sims = np.array(similarities)
    labs = np.array(labels, dtype=bool)
    predictions = sims >= threshold

    tp = np.sum(predictions & labs)
    tn = np.sum(~predictions & ~labs)
    fp = np.sum(predictions & ~labs)
    fn = np.sum(~predictions & labs)

    accuracy = (tp + tn) / len(labs) if len(labs) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_roc_curve(similarities, labels, n_thresholds=100):
    """Compute ROC curve (FPR, TPR) across thresholds from 0 to 100."""
    sims = np.array(similarities)
    labs = np.array(labels, dtype=bool)
    thresholds = np.linspace(0, 100, n_thresholds)

    fpr_list = []
    tpr_list = []

    for t in thresholds:
        predictions = sims >= t
        tp = np.sum(predictions & labs)
        fp = np.sum(predictions & ~labs)
        fn = np.sum(~predictions & labs)
        tn = np.sum(~predictions & ~labs)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def compute_auc(fpr, tpr):
    """Compute area under ROC curve using trapezoidal rule."""
    sorted_idx = np.argsort(fpr)
    return float(np.trapz(tpr[sorted_idx], fpr[sorted_idx]))
