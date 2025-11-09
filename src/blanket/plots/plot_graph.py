"""
Plotting functions for graphs.

#TODO: Using pydot
"""

from typing import Any

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, to_rgb
from matplotlib.lines import Line2D

from blanket.graph import markov_blanket as compute_markov_blanket
from blanket.utils import validate_adjmat

# Color scheme for graph visualization
COLOR_TARGET = "#FF6B6B"  # Red for target node
COLOR_MARKOV_BLANKET = "#FFA500"  # Orange for MB nodes
COLOR_OTHER = "#ADD8E6"  # Light blue for other nodes


def center_layout(
    G: nx.Graph,
    target_idx: int,
    center: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """Position nodes with target at center and others in a circle around it.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.
    target_idx : int
        Index of the target node to place at center.
    center : tuple of float, default (0.0, 0.0)
        Coordinate pair for the center position.

    Returns
    -------
    dict
        A dictionary of positions keyed by node.
    """
    nodes = list(G)

    if len(nodes) == 0:
        return {}

    # Place target at center
    pos = {target_idx: np.array(center)}

    # Get other nodes
    other_nodes = [n for n in nodes if n != target_idx]

    if len(other_nodes) > 0:
        # Place other nodes in a circle around the target
        theta = np.linspace(0, 1, len(other_nodes) + 1)[:-1] * 2 * np.pi
        other_pos = np.column_stack([np.cos(theta), np.sin(theta)])
        other_pos = other_pos + np.array(center)

        for node, coords in zip(other_nodes, other_pos):
            pos[node] = coords

    return pos


# TODO: when target_idx is none, treat it as a normal dag
def plot_graph(
    adjmat: np.ndarray,
    target_idx: int | None = None,
    markov_blanket: np.ndarray | None = None,
    max_nodes: int = 100,
    figsize: tuple = (5, 5),
    title: str | None = None,
    ax: Any | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Plot a graph from adjacency matrix with optional MB highlighting.

    Parameters
    ----------
    adjmat : np.ndarray
        Adjacency matrix of shape (n, n) representing the graph.
        The matrix must contain only 0s and 1s.
    target_idx : int or None, default None
        Index of the target node to highlight and center.
        If None, uses the last node as target.
    markov_blanket : np.ndarray or None, default None
        A MB mask of shape (n_nodes,) with 1s for nodes in the Markov blanket.
        If None, it will be computed automatically based on target_idx.
    max_nodes : int, default 100
        Maximum number of nodes to plot. If adjmat has more nodes,
        raises an error to prevent performance issues.
    figsize : tuple, default (5, 5)
        Figure size in inches (width, height).
    title : str or None, default None
        Custom title for the plot. If None, uses default title.
    ax : plt.Axes or None, default None
        Matplotlib axes to plot on. If None, creates a new figure.
    **kwargs : Any
        Additional keyword arguments passed to nx.draw_networkx functions
        for customization (e.g., node_size, width, etc.).

    Returns
    -------
    tuple of (plt.Figure, plt.Axes)
        The figure and axes objects containing the plot.

    Raises
    ------
    ValueError
        If adjmat is not square or has too many nodes.
    """
    # Validate input
    adjmat = validate_adjmat(adjmat)

    n_nodes = adjmat.shape[0]
    if n_nodes > max_nodes:
        raise ValueError(
            f"Graph has {n_nodes} nodes, exceeds max_nodes={max_nodes}. "
            "Use a larger max_nodes to override."
        )

    # Default to last node as target if not provided
    target: int = target_idx if target_idx is not None else n_nodes - 1

    # Compute Markov blanket if not provided
    mb: np.ndarray
    if markov_blanket is None:
        mb = compute_markov_blanket(adjmat, target)
    else:
        mb = markov_blanket

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create networkx graph from adjacency matrix
    G = nx.from_numpy_array(adjmat, create_using=nx.DiGraph)

    # Compute layout with target at center
    pos = center_layout(G, target_idx=target)

    # Determine node colors
    node_colors = [COLOR_OTHER] * n_nodes
    node_colors[target] = COLOR_TARGET
    for i, val in enumerate(mb):
        if val:
            node_colors[i] = COLOR_MARKOV_BLANKET

    # Set default drawing parameters
    draw_kwargs = {
        "node_color": node_colors,
        "node_size": 800,
        "edge_color": "gray",
        "arrows": True,
        "arrowsize": 20,
        "arrowstyle": "-|>",
        "connectionstyle": "arc3,rad=0.1",
        "width": 1.5,
    }
    # Update with any user-provided kwargs
    draw_kwargs.update(kwargs)

    # Draw the graph
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        **{k: v for k, v in draw_kwargs.items() if k in ["node_color", "node_size"]},
    )

    nx.draw_networkx_labels(G, pos, font_size=15, ax=ax)

    # Draw edges with user-customizable parameters
    edge_kwargs = {
        k: v
        for k, v in draw_kwargs.items()
        if k
        in [
            "edge_color",
            "arrows",
            "arrowsize",
            "arrowstyle",
            "connectionstyle",
            "width",
        ]
    }
    nx.draw_networkx_edges(G, pos, ax=ax, **edge_kwargs)

    # Create legend with circle markers
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=to_rgb(COLOR_TARGET),
            markersize=10,
            label="Target",
            markeredgecolor="black",
            markeredgewidth=0.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=to_rgb(COLOR_MARKOV_BLANKET),
            markersize=10,
            label="Markov Blanket",
            markeredgecolor="black",
            markeredgewidth=0.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=to_rgb(COLOR_OTHER),
            markersize=10,
            label="Others",
            markeredgecolor="black",
            markeredgewidth=0.5,
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

    # Set title
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    return fig, ax


def plot_adjmat(
    adjmat: np.ndarray,
    figsize: tuple = (5, 5),
    title: str | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot adjacency matrix as a heatmap with MB highlighting.

    Parameters
    ----------
    adjmat : np.ndarray
        Adjacency matrix of shape (n, n) representing the graph.
        The matrix must contain only 0s and 1s.
    figsize : tuple, default (5, 5)
        Figure size in inches (width, height).
    title : str or None, default None
        Custom title for the plot. If None, uses default title.
    ax : plt.Axes or None, default None
        Matplotlib axes to plot on. If None, creates a new figure.

    Returns
    -------
    tuple of (plt.Figure, plt.Axes)
        The figure and axes objects containing the plot.

    Raises
    ------
    ValueError
        If adjmat is not square.
    """
    # Validate input
    # INFO: allow adjacency matrices with non-binary values
    # adjmat = validate_adjmat(adjmat)

    n_nodes = adjmat.shape[0]

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot heatmap with discrete colorbar anchored on observed values
    valid_vals = adjmat[np.isfinite(adjmat)]
    if valid_vals.size == 0:
        valid_vals = np.array([0.0])
    unique_vals = np.unique(valid_vals)
    cmap = plt.get_cmap("binary", max(unique_vals.size, 2))
    if unique_vals.size == 1:
        step = abs(unique_vals[0]) * 0.5
        if step == 0:
            step = 0.5
        boundaries = np.array([unique_vals[0] - step, unique_vals[0] + step])
    else:
        min_diff = np.min(np.diff(unique_vals))
        step = min_diff / 2
        boundaries = np.concatenate(
            (
                [unique_vals[0] - step],
                (unique_vals[:-1] + unique_vals[1:]) / 2,
                [unique_vals[-1] + step],
            )
        )
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    im = ax.imshow(adjmat, cmap=cmap, norm=norm, aspect="equal")

    # Set square cells with light grey grid
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n_nodes), minor=False)
    ax.set_yticks(np.arange(n_nodes), minor=False)
    ax.set_xticklabels(np.arange(n_nodes), fontsize=10)
    ax.set_yticklabels(np.arange(n_nodes), fontsize=10)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, n_nodes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_nodes, 1), minor=True)
    ax.grid(which="minor", color="lightgrey", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
    ax.set_xlabel("To Node", fontsize=12, fontweight="bold")
    ax.set_ylabel("From Node", fontsize=12, fontweight="bold")

    # Add discrete colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=unique_vals, boundaries=boundaries, shrink=0.6)
    cbar.ax.tick_params(labelsize=10)

    # Set title
    ax.set_title(title, fontsize=14, fontweight="bold")

    return fig, ax
