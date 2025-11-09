"""Metrics for graph structure evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

from blanket.utils.validation import validate_adjmat


# TODO: validate NHD results
def shd(
    adj_true: np.ndarray, adj_pred: np.ndarray, *, normalized: bool = False
) -> float | int:
    """Compute SHD or normalized SHD (NHD) between two adjacency matrices.

    Parameters
    ----------
    adj_true : np.ndarray
        True adjacency matrix.
    adj_pred : np.ndarray
        Predicted adjacency matrix.
    normalized : bool, default=False
        If True, returns normalized SHD (NHD).

    Returns
    -------
    float | int
        SHD value (int) or normalized SHD (float).
    """
    adj_true = validate_adjmat(adj_true)
    adj_pred = validate_adjmat(adj_pred)

    if adj_true.shape != adj_pred.shape:
        raise ValueError("adj_true and adj_pred must have the same shape.")

    shd_value = int(np.count_nonzero(adj_true != adj_pred))
    n_nodes = adj_true.shape[0]
    max_edges = n_nodes * (n_nodes - 1) / 2
    nhd = shd_value / max_edges if max_edges else 0.0

    return nhd if normalized else shd_value


def adjacency_confusion(
    adj_true: np.ndarray, adj_pred: np.ndarray
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score for adjacency prediction.

    Parameters
    ----------
    adj_true : np.ndarray
        True adjacency matrix.
    adj_pred : np.ndarray
        Predicted adjacency matrix.

    Returns
    -------
    tuple[float, float, float]
        Precision, recall, and F1 score.
    """
    adj_true = validate_adjmat(adj_true)
    adj_pred = validate_adjmat(adj_pred)

    if adj_true.shape != adj_pred.shape:
        raise ValueError("adj_true and adj_pred must have the same shape.")

    true_flat = adj_true.ravel()
    pred_flat = adj_pred.ravel()

    cm = confusion_matrix(true_flat, pred_flat, labels=[0, 1])
    _, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )

    return precision, recall, f1
