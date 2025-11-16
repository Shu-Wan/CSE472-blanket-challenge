"""
Plotting functions for data distributions.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA


def reduce_dimensions(
    X_1: np.ndarray,
    X_2: np.ndarray,
    method: str = "umap",
    **kwargs: Any,
) -> np.ndarray:
    """Reduce dimensionality of combined distributions to 2D.

    Parameters
    ----------
    X_1 : np.ndarray
        First distribution (n_samples_1, n_features).
    X_2 : np.ndarray
        Second distribution (n_samples_2, n_features).
    method : str, default "umap"
        Dimensionality reduction method: "umap" or "pca".
    **kwargs : Any
        Additional keyword arguments passed to the specific method.

    Returns
    -------
    np.ndarray
        2D projection of combined data (n_samples_1 + n_samples_2, 2).
    """
    # Combine data
    X_combined = np.vstack([X_1, X_2])

    if method.lower() == "umap":
        # Set default UMAP parameters
        umap_params = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
        }
        umap_params.update(kwargs)
        reducer = umap.UMAP(**umap_params)
        X_proj = reducer.fit_transform(X_combined)

    elif method.lower() == "pca":
        # Set default PCA parameters
        pca_params = {"n_components": 2}
        pca_params.update(kwargs)
        reducer = PCA(**pca_params)
        X_proj = reducer.fit_transform(X_combined)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'pca'.")

    return X_proj


def plot_distributions(
    X_1: np.ndarray,
    X_2: np.ndarray,
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (5, 5),
    ax: Any = None,
    alpha: float = 0.6,
    s: int = 20,
    method: str = "umap",
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Plot two distributions using dimensionality reduction.

    Combines two ndarrays, applies dimensionality reduction (UMAP or PCA) for 2D projection,
    and visualizes them as a scatterplot with different colors for each distribution.

    Parameters
    ----------
    X_1 : np.ndarray
        First distribution (n_samples_1, n_features).
    X_2 : np.ndarray
        Second distribution (n_samples_2, n_features).
    labels : tuple[str, str] | None, default None
        Labels for the two distributions. If None, uses ["Distribution 1", "Distribution 2"].
    title : str | None, default None
        Title of the plot. If None, uses "Distribution Comparison ({method})".
    figsize : tuple[float, float], default (5, 5)
        Figure size (width, height).
    ax : Any, default None
        Matplotlib axes object. If None, creates a new figure.
    alpha : float, default 0.6
        Transparency level for scatter points.
    s : int, default 20
        Size of scatter points.
    method : str, default "umap"
        Dimensionality reduction method: "umap" or "pca".
    **kwargs : Any
        Additional keyword arguments passed to the specific method
        (e.g., n_neighbors, min_dist for UMAP; random_state for PCA).

    Returns
    -------
    tuple[Any, Any]
        The figure and axes objects.

    Examples
    --------
    >>> X_train = np.random.randn(100, 10)
    >>> X_test = np.random.randn(50, 10) + 0.5  # Shifted distribution
    >>> fig, ax = plot_distributions(
    ...     X_train, X_test,
    ...     labels=("Train", "Test"),
    ...     title="Train vs Test Distribution",
    ...     method="umap"
    ... )
    >>> plt.show()
    """

    if labels is None:
        labels = ("Distribution 1", "Distribution 2")

    if title is None:
        title = f"Distribution Comparison ({method.upper()})"

    # Get 2D projection
    n_1 = X_1.shape[0]
    X_proj = reduce_dimensions(X_1, X_2, method=method, **kwargs)

    # Create plot if axes not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create color array for both distributions
    colors = np.array(["#FF6B6B"] * n_1 + ["#4ECDC4"] * (X_proj.shape[0] - n_1))
    ax.scatter(
        X_proj[:, 0],
        X_proj[:, 1],
        alpha=alpha,
        s=s,
        c=colors,
    )

    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=12)

    # Add legend with color patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#FF6B6B", label=labels[0]),
        Patch(facecolor="#4ECDC4", label=labels[1]),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax
