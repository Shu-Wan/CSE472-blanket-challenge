"""Notears-based MB search."""

from typing import Any

import numpy as np
from castle.algorithms import Notears

from blanket.graph import markov_blanket

from ._utils import sanitize_cpdag


def notears_selector(
    X: np.ndarray,
    y: np.ndarray,
    lambda1: float = 0.1,
    loss_type: str = "l2",
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Notears algorithm for feature selection.

    Notears is a gradient-based algorithm for learning causal structures from
    continuous data. It uses an acyclicity constraint to ensure the learned
    structure forms a DAG (Directed Acyclic Graph).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target variable of shape (n_samples,) or (n_samples, 1).
    lambda1 : float, optional
        L1 penalty parameter for edge weights. Default: 0.1.
    loss_type : str, optional
        Loss function to use. Options:
        - 'l2': Least squares loss (default, for continuous data)
        - 'logistic': Logistic loss (for binary data)
        - 'poisson': Poisson loss (for count data)
        Default: 'l2'.
    max_iter : int, optional
        Maximum number of dual ascent steps. Default: 100.
    h_tol : float, optional
        Exit if acyclicity constraint violation <= h_tol. Default: 1e-8.
    rho_max : float, optional
        Exit if rho >= rho_max. Default: 1e16.
    w_threshold : float, optional
        Drop edge if |weight| < threshold. Default: 0.3.
    **kwargs : dict, optional
        Additional keyword arguments (reserved for future use).

    Returns
    -------
    feature_mask : np.ndarray
        Boolean mask array of shape (n_features,) indicating selected features.
    adjmat : np.ndarray
        Binary adjacency matrix of shape (n_features+1, n_features+1).

    Notes
    -----
    The target variable y is placed as the last node in the adjacency matrix.
    Notears is suitable for continuous data and uses gradient-based optimization.
    The lambda1 parameter controls sparsity - higher values lead to sparser graphs.

    References: https://github.com/huawei-noah/trustworthyAI/blob/master/gcastle/castle/algorithms/gradient/notears/linear.py

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1
    >>> feature_mask, adjmat = notears_selector(X, y, lambda1=0.1, loss_type='l2')
    >>> selected_features = np.where(feature_mask)[0]
    """
    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Concatenate X and y (y becomes the last column)
    data = np.hstack([X, y])

    # Run Notears algorithm
    nt = Notears(
        lambda1=lambda1,
        loss_type=loss_type,
        max_iter=max_iter,
        h_tol=h_tol,
        rho_max=rho_max,
        w_threshold=w_threshold,
    )
    nt.learn(data, **kwargs)
    cpdag = np.asarray(nt.causal_matrix, dtype=int)

    # Convert CPDAG to binary format
    adjmat = sanitize_cpdag(cpdag)

    # Target is the last node
    target_idx = X.shape[1]

    # Extract Markov blanket of the target
    feature_mask = markov_blanket(adjmat, target_idx)

    # Remove the target node from the mask to get only feature indices
    feature_mask = np.delete(feature_mask, target_idx)

    return feature_mask, adjmat
