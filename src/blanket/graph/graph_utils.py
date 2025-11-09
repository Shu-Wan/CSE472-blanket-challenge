from __future__ import annotations

import numpy as np

from blanket.utils.validation import validate_adjmat


def markov_blanket(adj_matrix: np.ndarray, target: int = -1) -> np.ndarray:
    """Compute the Markov blanket of a target node in a directed graph.

    The Markov blanket of a target node is the minimal set of nodes that
    renders the target conditionally independent from all other nodes. It
    comprises:
      - parents: nodes with edges into the target,
      - children: nodes with edges out of the target,
      - spouses: other parents (co-parents) of the target's children (excluding the target).

    Parameters
    ----------
    adj_matrix : numpy.ndarray
        Adjacency matrix of shape (n, n) where n is the number of nodes.
        adj_matrix[i, j] = 1 if there is an edge from i to j, 0 otherwise.
    target : int
        Index of the target node (0-based). Defaults to the last node.

    Returns
    -------
    numpy.ndarray
        Array of shape (n,) with 1 indicating nodes in the Markov blanket
        and 0 otherwise. The target node is the `target`-th value of the array.

    Examples
    --------
    >>> import numpy as np
    >>> adj = np.array([[0, 1, 0],
    ...                 [0, 0, 1],
    ...                 [1, 0, 0]])
    >>> markov_blanket(adj, 1)
    array([1, 0, 1])
    """
    adj_matrix = validate_adjmat(adj_matrix)
    n = adj_matrix.shape[0]

    # Parents: column vector for target
    parents = adj_matrix[:, target]

    # Children: row vector for target
    children = adj_matrix[target, :]

    # Spouses: co-parents of children, excluding target
    children_indices = np.where(children)[0]
    if len(children_indices) > 0:
        spouses = np.any(adj_matrix[:, children_indices], axis=1).astype(int)
    else:
        spouses = np.zeros(n, dtype=int)
    spouses[target] = 0  # exclude target

    # Combine and exclude target
    mb = parents | children | spouses
    mb[target] = 0

    return mb
