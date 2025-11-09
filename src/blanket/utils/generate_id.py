"""Generate id based on context."""

import hashlib
from typing import List, Union


def generate_id(prefix: str, context: Union[str, List[str]], length: int = 8) -> str:
    """Generate a deterministic ID from prefix and context.

    Args:
        prefix: Prefix for the ID (e.g., "graph")
        context: List of strings that determine the ID
        length: Length of the hash suffix (default: 8)

    Returns:
        Generated ID in format "{prefix}_{hash}"

    Example:
        >>> generate_id("graph", ["ER", "n=10", "p=0.3", "seed=42"])
        "graph_a1b2c3d4"
    """
    if isinstance(context, str):
        parts = [context]
    else:
        parts = list(map(str, context))
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"
