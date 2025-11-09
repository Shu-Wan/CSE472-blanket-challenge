"""Validation utilities for the blanket package."""

from __future__ import annotations

import numpy as np


def validate_adjmat(adj: np.ndarray) -> np.ndarray:
    """Ensure adjacency matrices are square, numeric, and binary.

    Parameters
    ----------
    adj : np.ndarray
        Input adjacency matrix.

    Returns
    -------
    np.ndarray
        Validated adjacency matrix (as int type).

    Raises
    ------
    ValueError
        If the matrix is not square or not binary.
    TypeError
        If the matrix is not numeric.
    """
    if not isinstance(adj, np.ndarray):
        adj = np.asarray(adj)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be a square 2D numpy array.")

    if not np.issubdtype(adj.dtype, np.number):
        raise TypeError("Adjacency matrix must contain numeric values.")

    unique_vals = np.unique(adj)
    if not np.all(np.isin(unique_vals, (0, 1))):
        raise ValueError(
            f"Adjacency matrix must only contain 0s and 1s. Found: {unique_vals}."
        )

    return adj.astype(int, copy=False)
