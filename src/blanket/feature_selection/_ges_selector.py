"""GES-based MB search."""

from typing import Any

import numpy as np
from castle.algorithms import GES

from blanket.graph import markov_blanket

from ._utils import sanitize_cpdag


def ges_selector(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "bic",
    method: str = "scatter",
    k: float = 0.001,
    N: int = 10,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GES algorithm for feature selection.

    Greedy Equivalence Search (GES) is a score-based causal discovery algorithm
    that searches for the graph with the best fit to the data according to a
    scoring criterion (e.g., BIC).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target variable of shape (n_samples,) or (n_samples, 1).
    criterion : str, optional
        Scoring criterion to use. Options:
        - 'bic': Bayesian Information Criterion (default)
        - 'bdeu': BDeu score (for discrete data)
        Default: 'bic'.
    method : str, optional
        Method for BIC scoring. Options:
        - 'scatter': Scatter matrix method (default)
        - 'r2': R-squared method
        Default: 'scatter'.
    k : float, optional
        Structure prior parameter for BDeu. Default: 0.001.
    N : int, optional
        Prior equivalent sample size for BDeu. Default: 10.
    **kwargs : dict, optional
        Additional keyword arguments passed to GES algorithm.

    Returns
    -------
    feature_mask : np.ndarray
        Boolean mask array of shape (n_features,) indicating selected features.
    adjmat : np.ndarray
        Binary adjacency matrix of shape (n_features+1, n_features+1).

    Notes
    -----
    The target variable y is placed as the last node in the adjacency matrix.
    GES returns a CPDAG which is converted to binary format before Markov
    blanket extraction.

    References: https://github.com/huawei-noah/trustworthyAI/blob/master/gcastle/castle/algorithms/ges/ges.py

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1
    >>> feature_mask, adjmat = ges_selector(X, y, criterion='bic', method='scatter')
    >>> selected_features = np.where(feature_mask)[0]
    """
    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Concatenate X and y (y becomes the last column)
    data = np.hstack([X, y])

    # Run GES algorithm
    ges = GES(criterion=criterion, method=method, k=k, N=N)
    ges.learn(data, **kwargs)
    cpdag = np.asarray(ges.causal_matrix, dtype=int)

    # Convert CPDAG to binary format
    adjmat = sanitize_cpdag(cpdag)

    # Extract Markov blanket of the target
    feature_mask = markov_blanket(adjmat, -1)

    # Remove the target node from the mask to get only feature indices
    feature_mask = np.delete(feature_mask, -1)

    return feature_mask, adjmat
