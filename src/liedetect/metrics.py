"""Evaluation metrics used in the thesis: AUC, MAP and BRP90."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_recall_at_precision(y_true, scores, target_precision: float = 0.90) -> float:
    """Highest recall reachable while keeping precision >= target_precision."""
    precisions, recalls, _ = precision_recall_curve(y_true, scores)
    valid = precisions >= target_precision
    return float(recalls[valid].max()) if valid.any() else 0.0


def compute_metrics(y_true, scores) -> dict:
    """Higher score must mean 'true statement' (label 1)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return {"AUC": np.nan, "MAP": np.nan, "BRP90": np.nan}
    return {
        "AUC": roc_auc_score(y_true, scores),
        "MAP": average_precision_score(y_true, scores),
        "BRP90": best_recall_at_precision(y_true, scores),
    }