"""Metrics for regression tasks."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        Mean squared error.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        Mean absolute error.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination) score.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        R² score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
