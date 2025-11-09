"""Graph generator with parallel generation support.

This module provides functions for generating directed acyclic graphs (DAGs)
as binary adjacency matrices. Supported methods include Erdős–Rényi (ER),
Preferential Attachment (PA), and Scale-Free networks (Price).

The GraphGenerator class supports parallel generation of multiple graphs
and uses density-based parameterization for ease of use.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

from blanket.utils import generate_id, get_numpy_rng

from .generator_utils import translate_params
from .graph import DAG

GRAPH_FEATURES = Features(
    {
        "graph_id": Value("string"),
        "adjacency_matrix": Sequence(Sequence(Value("uint8"))),
        "num_nodes": Value("int32"),
        "num_edges": Value("int32"),
        "density": Value("float32"),
        "method": Value("string"),
        "created_at": Value("string"),
        "seed": Value("int64"),
    }
)


class GraphGenerator:
    """Graph generator with parallel generation support.

    This class generates multiple graphs in parallel using density-based
    parameterization. It automatically translates (num_nodes, density) to
    method-specific parameters.

    Supports grid generation over methods, num_nodes, and density combinations.

    Example:
        # Single parameter set, sequential generation
        gen = GraphGenerator(
            method="ER",
            num_nodes=10,
            density=0.3,
            num_graphs=100,
            seed=42
        )
        dags = gen.generate()

        # Grid generation with parallelism
        gen = GraphGenerator(
            method=["ER", "PA"],
            num_nodes=[10, 20],
            density=[0.3, 0.5],
            num_graphs=50,
            seed=42,
            num_proc=4
        )
        dags = gen.generate()

    Attributes:
        method: Generation method(s) ("ER", "PA", "Price" or list)
        num_nodes: Number of nodes (int or list)
        density: Target edge density [0, 1] (float or list)
        num_graphs: Number of graphs per parameter combination
        seed: Base random seed
        num_proc: Number of parallel workers (None for sequential)
    """

    def __init__(
        self,
        method: str | List[str],
        num_nodes: int | List[int],
        density: float | List[float],
        num_graphs: int,
        seed: int = 42,
        num_proc: Optional[int] = None,
    ):
        """Initialize graph generator.

        Args:
            method: Generation method(s) ("ER", "PA", "Price" or list)
            num_nodes: Number of nodes (int or list of ints)
            density: Target edge density [0, 1] (float or list of floats)
            num_graphs: Number of graphs per parameter combination
            seed: Base random seed
            num_proc: Number of parallel workers (None for sequential)
        """
        # Normalize inputs to lists
        self.methods = [method] if isinstance(method, str) else method
        self.num_nodes_list = [num_nodes] if isinstance(num_nodes, int) else num_nodes
        self.density_list = [density] if isinstance(density, (int, float)) else density
        self.num_graphs = num_graphs
        self.seed = seed
        self.num_proc = num_proc

        # Validate methods
        for m in self.methods:
            if m not in GENERATORS:
                raise ValueError(
                    f"Unknown method: {m}. Options: {list(GENERATORS.keys())}"
                )

        # Validate num_nodes
        for n in self.num_nodes_list:
            if n < 2:
                raise ValueError(f"num_nodes must be >= 2, got {n}")

        # Validate density
        for d in self.density_list:
            if not 0.0 <= d <= 1.0:
                raise ValueError(f"density must be in [0, 1], got {d}")

        self._dags: Optional[List[DAG]] = None

    def generate(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """Generate graphs based on parameter grid.

        Generates num_graphs for each combination of (method, num_nodes, density).
        Returns records in HuggingFace dataset format.

        Args:
            verbose: Show progress bar

        Returns:
            List of dictionaries matching GRAPH_FEATURES schema
        """
        # Build parameter grid using itertools.product
        param_grid = []
        for method, num_nodes, density in product(
            self.methods, self.num_nodes_list, self.density_list
        ):
            for graph_idx in range(self.num_graphs):
                seed = self.seed + hash((method, num_nodes, density, graph_idx)) % (
                    2**31
                )
                param_grid.append((method, num_nodes, density, seed))

        records = []

        if self.num_proc is None or self.num_proc == 1:
            # Sequential generation
            iterator = (
                tqdm(param_grid, desc="Generating graphs") if verbose else param_grid
            )
            for method, num_nodes, density, seed in iterator:
                dag = _generate_single_graph(method, num_nodes, density, seed)
                # Generate graph_id based on method, num_nodes, density, seed
                graph_id = generate_id("graph", [method, num_nodes, density, seed])
                record = {
                    "graph_id": graph_id,
                    "adjacency_matrix": dag.adjacency_matrix.tolist(),
                    "num_nodes": int(dag.num_nodes),
                    "num_edges": int(dag.num_edges),
                    "density": float(dag.density),
                    "method": method,
                    "created_at": datetime.now().isoformat() + "Z",
                    "seed": seed,
                }
                records.append(record)
        else:
            # Parallel generation
            with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
                futures = [
                    executor.submit(
                        _generate_single_graph, method, num_nodes, density, seed
                    )
                    for method, num_nodes, density, seed in param_grid
                ]

                iterator = (
                    tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Generating graphs",
                    )
                    if verbose
                    else as_completed(futures)
                )

                for i, future in enumerate(iterator):
                    dag = future.result()
                    method, num_nodes, density, seed = param_grid[i]
                    # Generate graph_id based on method, num_nodes, density, seed
                    graph_id = generate_id("graph", [method, num_nodes, density, seed])
                    record = {
                        "graph_id": graph_id,
                        "adjacency_matrix": dag.adjacency_matrix.tolist(),
                        "num_nodes": int(dag.num_nodes),
                        "num_edges": int(dag.num_edges),
                        "density": float(dag.density),
                        "method": method,
                        "created_at": datetime.now().isoformat() + "Z",
                        "seed": seed,
                    }
                    records.append(record)

        self._records = records
        return records

    def save(
        self,
        path: str | Path,
        **kwargs,
    ) -> Path:
        """Save generated graphs to jsonl format.

        Args:
            path: Path to a file
            **kwargs: Additional arguments passed to HuggingFace Dataset.to_json()

        Returns:
            Path where dataset was saved
        """
        if not hasattr(self, "_records") or not self._records:
            raise RuntimeError("No graphs generated yet. Call generate() first.")

        # make sure path ends with .jsonl
        assert str(path).endswith(".jsonl"), "Path must end with .jsonl extension"

        # Create HF dataset directly from records
        dataset = Dataset.from_list(self._records, features=GRAPH_FEATURES)

        # Save to disk
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Pass num_proc from self and kwargs to save_to_disk
        save_kwargs = kwargs.copy()
        if self.num_proc is not None:
            save_kwargs["num_proc"] = self.num_proc

        dataset.to_json(str(path), **save_kwargs)

        return path


def _generate_single_graph(
    method: str,
    num_nodes: int,
    density: float,
    seed: int,
) -> DAG:
    """Generate a single graph (worker function for parallel generation).

    Args:
        method: Generation method ("ER", "PA", "Price")
        num_nodes: Number of nodes
        density: Target edge density
        seed: Random seed

    Returns:
        Generated DAG object
    """
    # Translate density to method-specific params
    params = translate_params(method, num_nodes, density, verbose=False)

    # Get generator function
    generator = GENERATORS[method]

    # Generate graph - generators now return DAG objects directly
    dag = generator(**params, seed=seed)

    return dag


def er_gnp(n: int, p: float, seed: int) -> DAG:
    """Generate a DAG from an Erdős–Rényi (G(n, p)) model.

    This function first samples an undirected simple graph G(n, p) using the
    provided seed, then orients each undirected edge according to a uniform
    random permutation of the node ordering. The resulting directed graph has
    no cycles by construction because all edges point from earlier to later
    nodes in the sampled permutation.

    Args:
        n: Number of nodes.
        p: Edge probability (for the undirected skeleton).
        seed: Random seed for reproducibility.

    Returns:
        DAG object
    """

    rng = get_numpy_rng(seed)

    # sample undirected edges only on the upper triangle
    iu, iv = np.triu_indices(n, k=1)
    U = rng.random(iu.size) < p  # boolean mask of present undirected edges

    # random topological order; orient edges from earlier -> later
    order = rng.permutation(n)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    u, v = iu[U], iv[U]  # select the present undirected edges
    forward = rank[u] < rank[v]  # decide direction per edge

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.uint8)
    for ui, vi, is_forward in zip(u, v, forward):
        if is_forward:
            A[ui, vi] = 1
        else:
            A[vi, ui] = 1

    return DAG(adjacency_matrix=A, num_nodes=n)


def pa_ba(n: int, m: int, seed: int) -> DAG:
    """Generate a DAG based on a Barabási–Albert preferential-attachment model.

    The undirected preferential-attachment graph is built incrementally: each
    new node attaches to `m` existing nodes with probability proportional
    to their degree. UNDIRECTED skeleton with a CLIQUE seed,
    then oriented by a random permutation (DAG).

    Args:
        n: Number of nodes.
        m: Number of edges each new node attaches with (must satisfy
            1 <= m < n).
        seed: Random seed for reproducibility.

    Returns:
        DAG object
    """

    if not (1 <= m < n):
        raise ValueError("m must be in [1, n-1).")

    rng_graph = get_numpy_rng(seed)
    rng_orient = get_numpy_rng((seed + 1) % (2**32) if seed is not None else None)

    A_und = np.zeros((n, n), dtype=bool)
    deg = np.zeros(n, dtype=np.int64)

    # clique seed on nodes [0..m-1]
    if m > 1:
        iu0, iv0 = np.triu_indices(m, k=1)
        A_und[iu0, iv0] = True
        A_und[iv0, iu0] = True
        deg[:m] = m - 1
    else:
        # m == 1 -> seed is a single isolated node (no edges yet)
        deg[0] = 0

    # repeated-nodes list (classic BA trick): node i appears deg[i] times
    repeated_nodes = [i for i in range(m) for _ in range(int(deg[i]))]

    # add remaining nodes
    for src in range(m, n):
        targets = []
        # sample m unique targets proportional to degree
        while len(targets) < m:
            # if repeated_nodes is empty (can only happen when m == 1 early on),
            # pick uniformly among existing nodes [0..src-1]
            if not repeated_nodes:
                t = int(rng_graph.integers(0, src))
            else:
                t = repeated_nodes[int(rng_graph.integers(0, len(repeated_nodes)))]
            if t not in targets:
                targets.append(t)

        for t in targets:
            if not A_und[src, t]:
                A_und[src, t] = A_und[t, src] = True
                deg[src] += 1
                deg[t] += 1
                repeated_nodes.append(t)  # one ticket for t's new edge
        repeated_nodes.extend([src] * m)  # m tickets for the new node

    # orientation based on random permutation
    order = rng_orient.permutation(n)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    iu, iv = np.triu_indices(n, k=1)
    has_edge = A_und[iu, iv]
    u, v = iu[has_edge], iv[has_edge]

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.uint8)
    forward = rank[u] < rank[v]
    for ui, vi, is_forward in zip(u, v, forward):
        if is_forward:
            A[ui, vi] = 1
        else:
            A[vi, ui] = 1

    return DAG(adjacency_matrix=A, num_nodes=n)


def pa_price(
    n: int,
    m_out: int,
    seed: int,
    m0: int | None = None,
) -> DAG:
    """Generate a DAG using Price's model (directed attachment with fixed out-degree).

    This procedure constructs a directed acyclic graph directly. Each new
    vertex chooses `m_out` existing vertices to point to. Attachment is biased
    by the current in-degree of vertices (preferential attachment). The
    parameter `m0` controls the size of the initial core.

    Args:
        n: Number of nodes.
        m_out: Number of outgoing arcs each new node creates.
        seed: Random seed for reproducibility.
        m0: Size of the initial seed core (defaults to m_out + 1).

    Returns:
        DAG object
    """

    if not (1 <= m_out < n):
        raise ValueError("m_out must be in [1, n-1).")

    m0 = m0 or (m_out + 1)
    if not (1 <= m0 <= n):
        raise ValueError("m0 must be in [1, n].")

    rng = get_numpy_rng(seed)

    A = np.zeros((n, n), dtype=np.uint8)
    in_deg = np.zeros(n, dtype=np.int64)

    for v in range(1, m0):
        k = min(m_out, v)
        if v <= k:
            targets = list(range(v))
        else:
            targets = rng.choice(np.arange(v), size=k, replace=False).tolist()

        for t in targets:
            A[v, t] = 1
            in_deg[t] += 1

    for v in range(m0, n):
        weights = (in_deg[:v] + 1).astype(float)
        total = float(weights.sum())
        targets = []
        chosen = set()

        for _ in range(m_out):
            r = rng.random() * total
            acc = 0.0
            pick = None
            for i, w in enumerate(weights):
                if i in chosen:
                    continue
                acc += w
                if acc >= r:
                    pick = i
                    break
            if pick is None:
                pick = next(i for i in range(v) if i not in chosen)

            targets.append(pick)
            chosen.add(pick)
            total -= weights[pick]

        for t in targets:
            A[v, t] = 1
            in_deg[t] += 1

    return DAG(adjacency_matrix=A, num_nodes=n)


# Registry of supported generators
GENERATORS = {
    "ER": er_gnp,
    "PA": pa_ba,
    "Price": pa_price,
}


# Convenience function
def generate_dag(method: str, params: Dict[str, Any], seed: int) -> DAG:
    """Convenience function to generate a single DAG with explicit params.

    This function uses method-specific parameters directly (not density).

    Args:
        method: Name of the generation method.
        params: Parameters passed to the underlying generator function.
        seed: Random seed used for deterministic generation.

    Returns:
        A :class:`DAG` instance.
    """
    if method not in GENERATORS:
        raise ValueError(
            f"Unknown method: {method}. Options: {list(GENERATORS.keys())}"
        )

    generator = GENERATORS[method]
    dag = generator(**params, seed=seed)

    return dag
