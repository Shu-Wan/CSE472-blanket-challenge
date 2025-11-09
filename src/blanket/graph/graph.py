"""Core graph structures for the blanket package."""

from typing import List, Tuple

import networkx as nx
import numpy as np


class DAG:
    """A simple Directed Acyclic Graph (DAG) representation.

    Stores graph as an adjacency matrix.

    Attributes:
        adjacency_matrix: Binary adjacency matrix of shape (num_nodes, num_nodes)
        num_nodes: Number of nodes in the DAG
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray | None = None,
        num_nodes: int | None = None,
        edge_list: List[Tuple[int, int]] | List[List[int]] | None = None,
    ):
        """Initialize a DAG.

        Args:
            adjacency_matrix: Binary adjacency matrix of shape (num_nodes, num_nodes).
                Required if edge_list is not provided.
            num_nodes: Number of nodes in the DAG. Required.
            edge_list: Edge list as either:
                - List of tuples: [(src1, dst1), (src2, dst2), ...]
                - List of two lists: [[src1, src2, ...], [dst1, dst2, ...]]
                If provided, adjacency_matrix is ignored.
        """
        if num_nodes is None:
            raise ValueError("num_nodes is required")

        self.num_nodes = num_nodes

        # Convert edge_list to adjacency matrix if provided
        if edge_list is not None:
            # Convert from list of tuples to list of two lists if needed
            if (
                edge_list
                and isinstance(edge_list[0], (tuple, list))
                and len(edge_list[0]) == 2
            ):
                # Check if it's list of tuples format [(0,1), (0,2)]
                if all(isinstance(e, tuple) for e in edge_list):
                    src_list = [e[0] for e in edge_list]
                    dst_list = [e[1] for e in edge_list]
                    edge_list = [src_list, dst_list]
                # Check if it's already in the right format [[0,0], [1,2]]
                elif len(edge_list) == 2 and isinstance(edge_list[0], list):
                    pass  # Already in correct format

            # Create adjacency matrix from edge list
            adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
            if edge_list and len(edge_list) == 2:
                sources, destinations = edge_list[0], edge_list[1]
                if len(sources) > 0:
                    adjacency_matrix[sources, destinations] = 1
        elif adjacency_matrix is None:
            raise ValueError("Either adjacency_matrix or edge_list must be provided")

        if adjacency_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Adjacency matrix shape {adjacency_matrix.shape} "
                f"does not match num_nodes {num_nodes}"
            )

        self.adjacency_matrix = adjacency_matrix.astype(np.uint8)

        # Validate it's a DAG
        if not self.is_dag():
            raise ValueError(
                "Adjacency matrix does not form a valid DAG (contains cycles)"
            )

    @property
    def num_edges(self) -> int:
        """Number of edges in the DAG."""
        return int(np.sum(self.adjacency_matrix))

    @property
    def density(self) -> float:
        """Edge density of the DAG."""
        return compute_density(self.num_nodes, self.num_edges)

    def is_dag(self) -> bool:
        """Check if the graph is acyclic using networkx."""
        G = self.to_networkx()
        return nx.is_directed_acyclic_graph(G)

    def to_adjacency_matrix(self) -> np.ndarray:
        """Return the adjacency matrix.

        Returns:
            Binary adjacency matrix of shape (num_nodes, num_nodes)
        """
        return self.adjacency_matrix.copy()

    def to_networkx(self) -> nx.DiGraph:
        """Convert the DAG to a networkx.DiGraph object.

        Returns:
            NetworkX DiGraph with metadata
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_nodes))
        rows, cols = np.nonzero(self.adjacency_matrix)
        G.add_edges_from(zip(rows, cols))
        return G

    def __repr__(self) -> str:
        """String representation showing key stats."""
        return (
            f"DAG(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, density={self.density:.3f})"
        )

    def __str__(self) -> str:
        """Detailed string representation."""
        return (
            f"DAG(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, density={self.density:.3f}, "
            f"is_dag={self.is_dag()})"
        )


def compute_density(num_nodes: int, num_edges: int) -> float:
    """Calculate the density of a directed graph.

    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges

    Returns:
        Edge density as float in [0, 1]
    """
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2 to compute density.")
    max_edges = num_nodes * (num_nodes - 1)
    return num_edges / max_edges if max_edges else 0.0
