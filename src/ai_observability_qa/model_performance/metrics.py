from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Calculate key classification performance metrics for QA validation.

    Args:
        y_true: True ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Optional predicted probabilities for the positive class (class 1).

    Returns:
        Dictionary of computed metric names and float values.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} items, but y_pred has {len(y_pred_arr)} items."
        )

    if len(y_true_arr) == 0:
        raise ValueError("Cannot calculate metrics on empty input arrays.")

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Confusion matrix extraction (binary classification)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    false_negative_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics: dict[str, float] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
        "total_samples": float(len(y_true_arr)),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        if len(y_prob_arr) != len(y_true_arr):
            raise ValueError(
                f"Length mismatch: y_prob has {len(y_prob_arr)} items, but y_true has {len(y_true_arr)} items."
            )
        # ROC-AUC is defined only when y_true has both classes
        unique_classes = np.unique(y_true_arr)
        if len(unique_classes) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
            # Clip probabilities slightly to avoid log(0) in log_loss
            eps = 1e-15
            clipped_probs = np.clip(y_prob_arr, eps, 1 - eps)
            metrics["log_loss"] = float(log_loss(y_true_arr, clipped_probs))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["log_loss"] = float("nan")

    return metrics


def calculate_sliced_metrics(
    dataframe: pd.DataFrame,
    slice_column: str,
    y_true_column: str,
    y_pred_column: str,
    y_prob_column: str | None = None,
) -> dict[Any, dict[str, float]]:
    """Calculate classification performance metrics per data slice/segment.

    Used by QA to uncover subgroup performance regressions (e.g. model
    underperforming on specific contract types or user demographics).
    """
    if slice_column not in dataframe.columns:
        raise ValueError(f"Slice column '{slice_column}' not found in dataframe.")
    if y_true_column not in dataframe.columns:
        raise ValueError(f"Ground truth column '{y_true_column}' not found in dataframe.")
    if y_pred_column not in dataframe.columns:
        raise ValueError(f"Prediction column '{y_pred_column}' not found in dataframe.")
    if y_prob_column and y_prob_column not in dataframe.columns:
        raise ValueError(f"Probability column '{y_prob_column}' not found in dataframe.")

    slice_results: dict[Any, dict[str, float]] = {}

    for slice_val, group in dataframe.groupby(slice_column):
        y_true = group[y_true_column]
        y_pred = group[y_pred_column]
        y_prob = group[y_prob_column] if y_prob_column else None

        slice_metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        slice_results[slice_val] = slice_metrics

    return slice_results

