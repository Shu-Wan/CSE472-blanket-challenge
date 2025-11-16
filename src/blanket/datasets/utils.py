import numpy as np

from blanket.utils.rng_control import get_numpy_rng


# TODO: select based on parents, spouses, children, markov blanket size count
def select_target_variable(num_nodes: int, seed: int = 42) -> int:
    """Select a random target variable.

    Args:
        num_nodes: Number of nodes in the graph
        seed: Random seed for reproducibility

    Returns:
        Index of target variable
    """
    rng = get_numpy_rng(seed)
    return int(rng.integers(0, num_nodes))


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    """Min-max scale scores to [0, 1].

    Args:
        scores: Array of scores to scale

    Returns:
        Scaled scores in [0, 1]
    """
    scores_min = scores.min()
    scores_max = scores.max()
    if scores_max - scores_min > 0:
        return (scores - scores_min) / (scores_max - scores_min)
    else:
        # All scores are the same, use uniform distribution
        return np.ones_like(scores) * 0.5
