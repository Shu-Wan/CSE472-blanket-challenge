"""Environment split functions for IID and distribution shift."""

from typing import Tuple

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA

from blanket.datasets.utils import minmax_scale
from blanket.utils.rng_control import get_numpy_rng


def create_iid_split(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create IID train/test split.

    Args:
        X: Feature matrix (n, d)
        y: Target vector (n,)
        train_fraction: Fraction of samples for training
        seed: Random seed

    Returns:
        Tuple of (train_idx, test_idx)
    """
    rng = get_numpy_rng(seed)
    n = len(X)

    # Random permutation
    indices = rng.permutation(n)

    # Split
    n_train = int(n * train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return train_idx, test_idx


def create_ood_split(
    X: np.ndarray,
    y: np.ndarray,
    shift_type: str = "covariate",
    projection: str = "random",
    feature_mask: np.ndarray | None = None,
    shift_mean: float = 0.3,
    shift_std: float = 0.2,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create OOD split via weighted sampling using Efraimidis-Spirakis algorithm.

    TODO:
        1. support other shift types (concept shift)
        2. support nonlinear projections (Kernal PCA, ICA)

    Args:
        X: Feature matrix (n, d)
        y: Target vector (n,)
        shift_type: "covariate" (covariate shift) or "label" (label shift)
        projection: "random", "pca"
        feature_mask: Optional boolean mask for selecting features (d,)
        shift_mean: Mean of shifting distribution
        shift_std: Std of shifting distribution
        train_fraction: Fraction of samples for training (α)
        seed: Random seed

    Returns:
        Tuple of (train_idx, test_idx)

    Implementation:
        1. Compute scores s (projected X or y)
        2. Normalize to [0, 1]
        3. Compute shifting weights w_i = norm(s_i; μ, σ)
        4. Apply ES algorithm: k_i = -ln(u_i) / w_i
        5. Select k samples with smallest keys for train
        6. Remaining samples go to test
    """
    n = len(X)
    k = int(n * train_fraction)

    # Step 1: Compute scores
    if shift_type == "covariate":
        if feature_mask is None:
            feature_mask = np.ones(X.shape[1], dtype=bool)
        scores = _linear_projection(X, projection, feature_mask, seed)
    elif shift_type == "label":
        # NOTE: also support nonlinear projections
        scores = y
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")

    # Step 2: Normalize scores to [0, 1]
    scores_normalized = minmax_scale(scores)

    # Step 3: Compute shifting weights
    weights = norm.pdf(scores_normalized, loc=shift_mean, scale=shift_std)

    # Step 4-5: Apply ES algorithm to select train samples
    train_idx = _efraimidis_spirakis_sample(weights, k, seed)

    # Step 6: Remaining samples go to test
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    return train_idx, test_idx


def _efraimidis_spirakis_sample(
    weights: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Efraimidis-Spirakis weighted random sampling (A-Res) with stable weights.

    Uses numerically stable formulation: k_i = -ln(u_i) / w_i
    Selects k samples with smallest keys.

    Args:
        weights: Sample weights (n,)
        k: Number of samples to select
        seed: Random seed

    Returns:
        Indices of selected samples (k,)

    References:
        [1] Efraimidis, Pavlos S., and Paul G. Spirakis. "Weighted random
        sampling with a reservoir." Information processing letters 97.5 (2006): 181-185.
    """
    rng = get_numpy_rng(seed)
    n = len(weights)

    # Generate uniform random variables
    u = rng.uniform(0, 1, n)

    # Compute keys: k_i = -ln(u_i) / w_i
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    keys = -np.log(u) / (weights + epsilon)

    # Select k samples with smallest keys
    selected_indices = np.argpartition(keys, k)[:k]

    return selected_indices


def _linear_projection(
    X: np.ndarray,
    method: str,
    feature_mask: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Project a vector to a scalar.

    Args:
        X: Feature matrix (n, d)
        method: "random", "pca"
        feature_mask: Optional boolean mask for selecting features (d,)
        seed: Random seed

    Returns:
        Projection scores (n,)
    """
    rng = get_numpy_rng(seed)
    n, d = X.shape

    if feature_mask is None:
        feature_mask = np.ones(X.shape[1], dtype=bool)

    X = X[:, feature_mask]

    if method == "random":
        # Random projection: w ~ N(0, I_d), normalize to ||w|| = 1
        w = rng.normal(0, 1, d)
        w = w / np.linalg.norm(w)
        scores = X @ w

    elif method == "pca":
        # First principal component
        pca = PCA(n_components=1, random_state=seed)
        scores = pca.fit_transform(X).flatten()

    else:
        raise ValueError(f"Unknown projection method: {method}")

    return scores
