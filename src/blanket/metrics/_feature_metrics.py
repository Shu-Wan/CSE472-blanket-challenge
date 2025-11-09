"""Metrics for feature selection evaluation."""

from __future__ import annotations

import numpy as np


def jaccard_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Jaccard similarity coefficient between two feature sets.

    The Jaccard score measures the similarity between two sets by computing
    the ratio of intersection to union.

    Jaccard score is symmetric, the order of inputs does not matter.
    For convention, we still name two inputs y_true and y_pred.

    Parameters
    ----------
    y_true : np.ndarray
        Boolean mask or indices of reference features (e.g., Markov blanket).
    y_pred : np.ndarray
        Boolean mask or indices of selected features.

    Returns
    -------
    float
        Jaccard similarity score in [0, 1].

    Examples
    --------
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_pred = np.array([1, 1, 0, 1, 0])
    >>> jaccard_score(y_true, y_pred)
    0.5
    """
    # Convert to boolean masks if needed
    selected_bool = np.asarray(y_pred, dtype=bool)
    reference_bool = np.asarray(y_true, dtype=bool)

    intersection = np.sum(selected_bool & reference_bool)
    union = np.sum(selected_bool | reference_bool)

    return float(intersection / union) if union > 0 else 0.0


def reduction_rate(y_pred: np.ndarray, total_features: int | None = None) -> float:
    """Calculate the proportion of features reduced (removed) from the original set.

    This measures how much the feature set was reduced during selection,
    complementing the selection rate.

    Parameters
    ----------
    y_pred : np.ndarray
        Boolean mask of selected features.
    total_features : int, optional
        Total number of features. If None, uses the length of y_pred.

    Returns
    -------
    float
        Proportion of features reduced in [0, 1].

    Examples
    --------
    >>> y_pred = np.array([1, 1, 0, 1, 0])  # 3 out of 5 selected
    >>> reduction_rate(y_pred)
    0.4  # 40% of features were reduced/removed
    """
    selected_bool = np.asarray(y_pred, dtype=bool)
    if total_features is None:
        total_features = len(selected_bool)

    n_selected = np.sum(selected_bool)
    n_reduced = total_features - n_selected
    return float(n_reduced / total_features) if total_features > 0 else 0.0
