"""Graph generation configuration."""

from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, field_validator


class GraphGenerationConfig(BaseModel):
    """Graph generation configuration.

    This configuration defines parameters for graph generation using GraphGenerator.
    GraphGenerator natively handles parameter expansion for lists of values.

    Attributes:
        method: Generation method(s) ("ER", "PA", "Price" or list)
        num_nodes: Number of nodes (int or list for grid search)
        density: Edge density (float or list for grid search)
        num_graphs: Number of graphs per parameter combination
        seed: Base random seed for reproducibility
        num_proc: Number of parallel workers (1 for sequential, >1 for parallel, -1 for all cores)
        path: Optional path to save generated graphs (.jsonl format)
    """

    method: Annotated[str | List[str], Field(description="Generation method(s)")]
    num_nodes: Annotated[int | List[int], Field(description="Number of nodes (>=2)")]
    density: Annotated[float | List[float], Field(description="Edge density [0, 1]")]

    num_graphs: Annotated[int, Field(ge=1)] = 5
    seed: int = 42
    num_proc: int = 1

    path: Optional[str] = None

    @field_validator("method", mode="after")
    @classmethod
    def _validate_methods(cls, v: str | List[str]) -> str | List[str]:
        """Validate methods against supported generators."""
        allowed = {"ER", "PA", "Price"}
        methods = [v] if isinstance(v, str) else v
        for method in methods:
            if method not in allowed:
                raise ValueError(
                    f"Unknown method: {method}. Allowed: {sorted(allowed)}"
                )
        return v

    @field_validator("num_nodes", mode="after")
    @classmethod
    def _validate_num_nodes(cls, v: int | List[int]) -> int | List[int]:
        """Validate num_nodes values."""
        values = [v] if isinstance(v, int) else v
        for val in values:
            if val < 2:
                raise ValueError(f"num_nodes must be >= 2, got {val}")
        return v

    @field_validator("density", mode="after")
    @classmethod
    def _validate_density(cls, v: float | List[float]) -> float | List[float]:
        """Validate density values."""
        values = [v] if isinstance(v, (int, float)) else v
        for val in values:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"density must be in [0, 1], got {val}")
        return v
