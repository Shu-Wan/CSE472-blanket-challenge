"""Structural Causal Model generation functions."""

import networkx as nx
import numpy as np

from blanket.utils import get_numpy_rng


def generate_linear_scm(
    adj_matrix: np.ndarray,
    n_samples: int,
    nonlinear: bool = False,
    coeff_range: float = 1.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Generate samples from linear Gaussian SCM.

    For each node in topological order:
        X_i = sum_j(beta_ij * X_j) + noise_i

    Args:
        adj_matrix: Adjacency matrix (n x n) where adj[i,j]=1 means edge i->j
        n_samples: Number of samples to generate
        nonlinear: If True, apply random nonlinear functions
        coeff_range: Range of coefficients: [-coeff_range, coeff_range]
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed

    Returns:
        Array of shape (n_samples, num_nodes)
    """
    rng = get_numpy_rng(seed)

    # Convert to NetworkX only for topological sort
    dag = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    n_nodes = adj_matrix.shape[0]
    X = np.zeros((n_samples, n_nodes))

    # Get topological order
    topo_order = list(nx.topological_sort(dag))

    for node in topo_order:
        # Get parents of this node (adj_matrix[:, node] gives column for node)
        parents = np.where(adj_matrix[:, node])[0]

        if len(parents) == 0:
            # Root node: pure noise
            X[:, node] = rng.normal(0, noise_std, n_samples)
        else:
            # Linear combination of parents
            parent_data = X[:, parents]  # shape: (n_samples, num_parents)

            # Sample random coefficients in [-coeff_range, coeff_range]
            coeffs = rng.uniform(-coeff_range, coeff_range, len(parents))

            # Linear combination
            linear_part = parent_data @ coeffs

            if nonlinear:
                # Apply random nonlinearity
                linear_part = apply_random_nonlinearity(linear_part, rng)

            # Add Gaussian noise
            noise = rng.normal(0, noise_std, n_samples)
            X[:, node] = linear_part + noise

    return X


def apply_random_nonlinearity(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random bounded nonlinear function (sin, cos, tanh, x^2, etc.).

    Args:
        x: Input data array
        rng: Random number generator

    Returns:
        Transformed data with nonlinear function applied (bounded to avoid overflow)
    """
    # Clip input to reasonable range to prevent overflow
    x_clipped = np.clip(x, -3, 3)

    # Choose random nonlinear function (all bounded to avoid numerical issues)
    functions = [
        np.sin,  # sine (bounded [-1, 1])
        np.cos,  # cosine (bounded [-1, 1])
        np.tanh,  # tanh (bounded [-1, 1])
        lambda x: 0.5 * np.square(x),  # scaled quadratic
        lambda x: x / (1 + np.abs(x)),  # smooth bounded function
    ]

    func_idx = rng.integers(0, len(functions))
    func = functions[func_idx]

    # Apply function with safe scaling
    result = func(x_clipped)
    # Ensure output is bounded
    result = np.clip(result, -5, 5)
    return result
