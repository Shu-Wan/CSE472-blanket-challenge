"""PC-based Markov Blanket Search"""

from typing import Any

import numpy as np
from castle.algorithms import PC
from castle.common.priori_knowledge import PrioriKnowledge

from blanket.graph import get_markov_blanket

from ._utils import sanitize_cpdag


def pc_selector(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    ci_test: str = "fisherz",
    variant: str = "stable",
    priori_knowledge: PrioriKnowledge = None,  # type: ignore
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PC algorithm for feature selection.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target variable of shape (n_samples,) or (n_samples, 1).
    alpha : float, optional
        Significance level for conditional independence tests (p-value threshold).
        Must be in (0, 1). Default: 0.05.
    ci_test : str, optional
        Name of the independence test to use. Options:
        - 'fisherz': Fisher's Z test (default, suitable for continuous data)
        - 'g2': G-squared test (for discrete data)
        - 'chi2': Chi-squared test (for discrete data)
        Default: 'fisherz'.
    variant : str, optional
        Variant of PC algorithm to use. Options:
        - 'original': Original PC algorithm
        - 'stable': Stable PC algorithm (default, recommended)
        - 'parallel': Parallel PC algorithm
        Default: 'stable'.
    priori_knowledge : PrioriKnowledge, optional
        Prior knowledge matrix for the algorithm. Default: None.
    **kwargs : dict, optional
        Advanced parameters passed to pc.learn():
        - p_cores : int - number of CPU cores to use
        - s : bool - memory-efficient indicator
        - batch : int - number of edges per batch

    Returns
    -------
    feature_mask : np.ndarray
        Boolean mask array of shape (n_features,) indicating selected features.
    adjmat : np.ndarray
        Binary adjacency matrix of shape (n_features+1, n_features+1).

    Notes
    -----
    The target variable y is placed as the last node in the adjacency matrix.
    The PC algorithm returns a CPDAG which is converted to binary format before
    Markov blanket extraction.

    Reference: https://github.com/huawei-noah/trustworthyAI/blob/master/gcastle/castle/algorithms/pc/pc.py

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1
    >>> feature_mask, adjmat = pc_selector(X, y, alpha=0.05, ci_test="fisherz", variant="stable")
    >>> selected_features = np.where(feature_mask)[0]
    """
    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # add y as the last column
    data = np.hstack([X, y])

    # Run PC algorithm
    pc = PC(
        variant=variant, alpha=alpha, ci_test=ci_test, priori_knowledge=priori_knowledge
    )
    pc.learn(data, **kwargs)
    cpdag = np.asarray(pc.causal_matrix, dtype=int)

    # Convert CPDAG to DAG by resolving bidirectional edges
    adjmat = sanitize_cpdag(cpdag)

    # Extract Markov blanket of the target
    feature_mask = get_markov_blanket(adjmat, -1)

    # Remove the target node from the mask to get only feature indices
    feature_mask = np.delete(feature_mask, -1)

    return feature_mask, adjmat
