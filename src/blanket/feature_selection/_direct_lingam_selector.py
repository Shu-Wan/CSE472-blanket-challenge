"""DirectLiNGAM-based MB search."""

from typing import Any

import numpy as np
from castle.algorithms import DirectLiNGAM

from blanket.graph import markov_blanket

from ._utils import sanitize_cpdag


def direct_lingam_selector(
    X: np.ndarray,
    y: np.ndarray,
    prior_knowledge: np.ndarray | None = None,
    measure: str = "pwling",
    thresh: float = 0.3,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run DirectLiNGAM algorithm for feature selection.

    DirectLiNGAM is a direct learning algorithm for linear non-Gaussian acyclic
    models (LiNGAM). It assumes that the data follows a linear model with
    non-Gaussian noise, which allows for the identification of the full
    causal structure including edge directions.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target variable of shape (n_samples,) or (n_samples, 1).
    prior_knowledge : np.ndarray | None, optional
        Prior knowledge matrix of shape (n_features+1, n_features+1).
        Elements are defined as:
        - 0: i does not have a directed path to j
        - 1: i has a directed path to j
        - -1: no prior knowledge available
        Default: None.
    measure : str, optional
        Measure to evaluate independence. Options:
        - 'pwling': Pairwise-likelihood ratio (default, better for general use)
        - 'kernel': Kernel-based measure
        Default: 'pwling'.
    thresh : float, optional
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
    DirectLiNGAM assumes non-Gaussian noise, which may not hold for all datasets.
    The algorithm is most suitable for continuous data with non-Gaussian distributions.

    References
    ----------
    Shimizu, S., Inazumi, T., Sogawa, Y., Hyvärinen, A., Kawahara, Y., Washio, T.,
    ... & Bollen, K. (2011). DirectLiNGAM: A direct method for learning a linear
    non-Gaussian structural equation model. Journal of Machine Learning Research,
    12(Apr), 1225-1248.

    Hyvärinen, A., & Smith, S. M. (2013). Pairwise likelihood ratios for estimation
    of non-Gaussian structural equation models. Journal of Machine Learning Research,
    14(111), 111-152.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1
    >>> feature_mask, adjmat = direct_lingam_selector(X, y, measure='pwling', thresh=0.3)
    >>> selected_features = np.where(feature_mask)[0]
    """
    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Concatenate X and y (y becomes the last column)
    data = np.hstack([X, y])

    # Run DirectLiNGAM algorithm
    dl = DirectLiNGAM(prior_knowledge=prior_knowledge, measure=measure, thresh=thresh)
    dl.learn(data, **kwargs)
    cpdag = np.asarray(dl.causal_matrix, dtype=int)

    # Convert CPDAG to binary format
    adjmat = sanitize_cpdag(cpdag)

    # Target is the last node
    target_idx = X.shape[1]

    # Extract Markov blanket of the target
    feature_mask = markov_blanket(adjmat, target_idx)

    # Remove the target node from the mask to get only feature indices
    feature_mask = np.delete(feature_mask, target_idx)

    return feature_mask, adjmat
