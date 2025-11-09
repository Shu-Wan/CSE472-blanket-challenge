"""translate num of nodes, density to generation methods params"""

import math
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

# Supported generators (imported in generator.py)
_GENERATORS = {"ER", "PA", "Price"}


def _density_floor(n: int, beta: float = 1.0) -> float:
    """
    Adaptive minimum density to avoid degenerate, ultra-sparse tiny graphs.

    The floor is the min of two principled terms:
      1) Quantization floor: beta * (2 / (n - 1))
         - Rationale: In PA/Price, achievable densities change in discrete steps of
           roughly Δρ ≈ 2/(n-1) as you vary the integer parameter m (or m_out).
           Requiring rho >= beta * 2/(n-1) means your target is at least `beta`
           *steps* away from zero, so rounding the integer parameter won't
           obliterate your intent.
      2) "At least a tree" floor: 2 / n
         - Rationale: Ensures the *expected* number of edges is ≥ (n-1), i.e.,
           we stay out of the trivial regime with lots of isolates.

    Combined:
        density_floor(n, beta) = min( beta * 2/(n-1),  2/n )

    Notes:
    - This is model-agnostic and only applied to user *requests* (translation).
    - For large n, both terms → 0, so the floor becomes negligible.
    """
    if n < 2:
        return 0.0
    quantization = beta * (2.0 / (n - 1))
    tree_like = 2.0 / n
    return min(quantization, tree_like)


def _edges_ba_price(n: int, m: int) -> int:
    """Exact undirected edge count for BA and Price (with m0 = m + 1): E = m*n - (m^2 + m)/2."""
    return m * n - (m * (m + 1)) // 2


def _achieved_density(n: int, m: int) -> float:
    if n < 2:
        return 0.0
    return (2.0 * _edges_ba_price(n, m)) / (n * (n - 1))


def _solve_m_from_density(n: int, rho: float) -> float:
    """
    Solve for real-valued m* from the exact expected-edges equation shared by PA and Price:
        m^2 - (2n-1)m + rho * n * (n-1) = 0
    Returns the smaller root (monotone in rho).
    """
    a = 1.0
    b = -(2.0 * n - 1.0)
    c = rho * n * (n - 1.0)
    disc = b * b - 4.0 * a * c
    if disc < 0 and disc > -1e-12:
        disc = 0.0  # clean tiny negative due to FP
    if disc < 0:
        raise ValueError(
            f"No real solution for parameters at density={rho:.4g}, n={n} "
            f"(discriminant={disc:.3g} < 0). Increase density."
        )
    return (-(b) - math.sqrt(disc)) / (2.0 * a)


def _pick_m_closest(n: int, m_star: float, rho_target: float) -> int:
    """
    Choose the integer m in [1, n-1] whose achieved density is *closest* to the target.
    This removes systematic over/under across the density range.
    """
    lo = max(1, min(n - 1, int(math.floor(m_star))))
    hi = max(1, min(n - 1, int(math.ceil(m_star))))
    dlo = abs(_achieved_density(n, lo) - rho_target)
    dhi = abs(_achieved_density(n, hi) - rho_target)
    return lo if dlo <= dhi else hi


def translate_params(
    method: str,
    num_nodes: int,
    density: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Translate (n, density) into parameters for {"ER","PA","Price"}.

    Design:
      - Solves the *exact* expected-edges equations.
      - Enforces an adaptive density floor to avoid degenerate tiny graphs:
          density >= max(beta*2/(n-1), 2/n)
        with beta=2 (two quantization steps above zero).
      - For PA and Price, picks the integer parameter (m or m_out) whose
        achieved density is *closest* to the target — fixing the systematic
        rounding bias you observed (PA below target when ρ>0.5, Price above
        target when ρ<0.5).

    Returns:
      ER    → {"n": n, "p": p}
      PA    → {"n": n, "m": m}
      Price → {"n": n, "m_out": m_out, "m0": m_out + 1}
    """
    n = int(num_nodes)
    if n < 2:
        raise ValueError("num_nodes must be >= 2")

    density = float(np.clip(density, 0.0, 1.0))

    # Adaptive sparsity guard
    floor_val = _density_floor(n, beta=2.0)
    if density < floor_val:
        raise ValueError(
            f"Requested density {density:.4g} is too sparse for n={n}. "
            f"Minimum recommended density is ~{floor_val:.4g} "
            "(quantization + tree floor)."
        )

    if method == "ER":
        p = density  # exact-in-expectation
        if verbose:
            print(f"ER: Target density = {density:.6f}, Achieved density = {p:.6f}")
        return {"n": n, "p": p}

    # Shared solve for PA and Price
    m_star = _solve_m_from_density(n, density)
    m_best = _pick_m_closest(n, m_star, density)
    achieved = _achieved_density(n, m_best)

    if method == "PA":
        if verbose:
            print(
                f"PA: Target density = {density:.6f}, "
                f"Achieved density = {achieved:.6f}, m = {m_best}"
            )
        return {"n": n, "m": m_best}

    if method == "Price":
        m_out = m_best
        m0 = min(n, m_out + 2)
        if verbose:
            print(
                f"Price: Target density = {density:.6f},",
                f" Achieved density = {achieved:.6f},",
                f" m_out = {m_out}, m0 = {m0}",
            )
        return {"n": n, "m_out": m_out, "m0": m0}

    raise ValueError(f"Unknown method: {method}. Options: {sorted(_GENERATORS)}")


def adjmat_to_edgelist(adj_matrix: NDArray[np.uint8]) -> List[List[int]]:
    """
    Convert a binary adjacency matrix to edge list format (source, destination).

    Args:
        adj_matrix: Square binary adjacency matrix (n x n)

    Returns:
        List of [source_nodes, dest_nodes] lists

    Example:
        >>> adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
        >>> adjmat_to_edgelist(adj)
        [[0, 1], [1, 2]]
    """
    sources, destinations = np.nonzero(adj_matrix)
    return [sources.tolist(), destinations.tolist()]


def edgelist_to_adjmat(edge_list: List[List[int]], num_nodes: int) -> NDArray[np.uint8]:
    """
    Convert edge list format to binary adjacency matrix.

    Args:
        edge_list: List of [source_nodes, dest_nodes] lists
        num_nodes: Number of nodes in the graph

    Returns:
        Square binary adjacency matrix (num_nodes x num_nodes)

    Example:
        >>> edge_list = [[0, 1], [1, 2]]
        >>> edgelist_to_adjmat(edge_list, 3)
        array([[0, 1, 0],
               [0, 0, 1],
               [0, 0, 0]], dtype=uint8)
    """
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
    sources, destinations = edge_list[0], edge_list[1]

    assert len(sources) == len(destinations), (
        "Edge index lists must be of equal length."
    )
    assert all(0 <= s < num_nodes for s in sources), (
        "Source node indices out of bounds."
    )
    assert all(0 <= d < num_nodes for d in destinations), (
        "Destination node indices out of bounds."
    )

    if len(sources) > 0:
        adj_matrix[sources, destinations] = 1
    return adj_matrix
