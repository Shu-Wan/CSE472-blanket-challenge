"""Utility functions for feature selection."""

import numpy as np


def sanitize_cpdag(cpdag: np.ndarray) -> np.ndarray:
    """Sanitize cpdag to a unified binary format.

    Converts different CPDAG formats to a unified binary representation:
    - cpdag[i, j] == 1 means there is an edge i -> j (or part of a
      bidirectional/undirected relation)
    - cpdag[i, j] == 0 means no edge

    Handles CPDAG from different causal discovery algorithms:

    1. causallearn format with -1, 0, 1:
       - cpdag[j,i]==1 and cpdag[i,j]==-1 indicates i -> j
       - cpdag[i,j]==cpdag[j,i]==-1 indicates i -- j (undirected)
       - cpdag[i,j]==cpdag[j,i]==1 indicates i <-> j (bidirectional)

    2. gcastle format with 0, 1:
       - cpdag[i,j]==1 and cpdag[j,i]==0 means i -> j
       - cpdag[i,j]==cpdag[j,i]==1 means i <-> j (bidirectional)
       - cpdag[i,j]==0 and cpdag[j,i]==0 means no edge

    Parameters
    ----------
    cpdag : np.ndarray
        CPDAG adjacency matrix from causal discovery algorithm.

    Returns
    -------
    np.ndarray
        Binary adjacency matrix with unified format.
    """

    cpdag = np.asarray(cpdag)

    # Already binary (e.g., gCastle output) -> return a copy
    if ((cpdag == 0) | (cpdag == 1)).all():
        return cpdag.astype(int, copy=True)

    # Masks for causallearn's {-1, 0, 1} encoding
    directed = (cpdag == -1) & (cpdag.T == 1)
    bidirectional = (cpdag == 1) & (cpdag.T == 1)
    undirected = (cpdag == -1) & (cpdag.T == -1)

    new_cpdag = np.zeros_like(cpdag, dtype=int)
    new_cpdag[directed] = 1
    new_cpdag[bidirectional | undirected] = 1

    return new_cpdag
