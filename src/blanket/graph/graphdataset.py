"""HuggingFace dataset schema and utilities for graph storage."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, Features, Sequence, Value, load_dataset, load_from_disk
import numpy as np

from .graph import DAG

# HuggingFace Dataset schema for graph storage
GRAPH_FEATURES = Features(
    {
        "graph_id": Value("string"),
        "adjacency_matrix": Sequence(Sequence(Value("uint8"))),
        "num_nodes": Value("int32"),
        "num_edges": Value("int32"),
        "density": Value("float32"),
        "method": Value("string"),
        "created_at": Value("string"),
        "seed": Value("int32"),
    }
)


class GraphDataset:
    """
    Graph Dataset with HuggingFace Hub integration.

    This class manages graph datasets, including loading from local paths
    or HuggingFace Hub, and pushing to the Hub.

    Attributes:
        dataset: The underlying HuggingFace Dataset
        metadata: Dataset metadata dictionary
    """

    def __init__(
        self,
        dataset: Dataset,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize GraphDataset.

        Args:
            dataset: HuggingFace Dataset object
            metadata: Optional metadata dictionary
        """
        self.dataset = dataset
        self.metadata = metadata or {}

    @classmethod
    def load(
        cls,
        path_or_repo: str,
        from_hub: bool = False,
        **kwargs,
    ):
        """Load dataset from local path or HuggingFace Hub.

        Args:
            path_or_repo: Local path or HuggingFace repo ID
            from_hub: If True, load from HuggingFace Hub
            **kwargs: Additional arguments for load_dataset

        Returns:
            GraphDataset instance
        """
        # Load metadata if it exists
        metadata = None

        if from_hub:
            # Load from HuggingFace Hub
            dataset = load_dataset(path_or_repo, **kwargs)
            # If DatasetDict is returned, get the first split
            if hasattr(dataset, "keys"):
                # Get the first available split
                split_name = list(dataset.keys())[0]
                dataset = dataset[split_name]
        else:
            # Load from local disk
            path = Path(path_or_repo)
            dataset = load_from_disk(str(path), **kwargs)

            # Try to load metadata if it exists
            metadata_path = path.parent / f"{path.name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

        return cls(dataset=dataset, metadata=metadata)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = True,
        commit_message: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Push dataset to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
            private: Whether the repository should be private
            commit_message: Optional commit message
            **kwargs: Additional arguments for push_to_hub
        """
        # Set default commit message if not provided
        if commit_message is None:
            commit_message = f"Upload dataset to {repo_id}"

        # Push the dataset to the hub
        self.dataset.push_to_hub(
            repo_id=repo_id, private=private, commit_message=commit_message, **kwargs
        )

    def save(
        self,
        path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save dataset to local disk with metadata.

        Args:
            path: Local path to save dataset
            metadata: Optional metadata to save alongside dataset

        Returns:
            Path where dataset was saved
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the dataset
        self.dataset.save_to_disk(str(path))

        # Save metadata if provided or if instance has metadata
        metadata_to_save = metadata or self.metadata
        if metadata_to_save:
            metadata_path = path.parent / f"{path.name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata_to_save, f, indent=2)

        return path


# Metadata utilities
def create_dataset_metadata(
    dataset_name: str,
    generator_config: Dict[str, Any],
    n_graphs: int,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create metadata dictionary for a graph dataset.

    Args:
        dataset_name: Name of the dataset
        generator_config: Configuration used for generation
        n_graphs: Number of graphs in the dataset
        notes: Optional notes

    Returns:
        Metadata dictionary
    """
    # Try to get git version
    try:
        import subprocess

        code_version = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]
        )
    except Exception:
        code_version = "unknown"

    config_hash = compute_config_hash(generator_config)

    metadata = {
        "project": "blanket",
        "dataset_name": dataset_name,
        "generator_config": generator_config,
        "config_hash": config_hash,
        "code_version": code_version,
        "export_timestamp": datetime.now().isoformat(),
        "n_graphs": n_graphs,
    }

    if notes:
        metadata["notes"] = notes

    # Add param grid summary if applicable
    if "methods" in generator_config:
        methods = generator_config.get("methods", [])
        num_nodes = generator_config.get("num_nodes", "?")
        density = generator_config.get("density", "?")
        metadata["param_grid_summary"] = (
            f"methods={methods}, num_nodes={num_nodes}, density={density}"
        )

    return metadata


def hf_record_to_dag(record: Dict[str, Any]) -> DAG:
    """
    Convert a HuggingFace dataset record back to a DAG.

    Args:
        record: Dictionary matching GRAPH_FEATURES schema

    Returns:
        Reconstructed DAG object
    """
    # Convert adjacency matrix from nested list to numpy array
    adj_matrix = np.array(record["adjacency_matrix"], dtype=np.uint8)

    return DAG(
        adjacency_matrix=adj_matrix,
        num_nodes=record["num_nodes"],
    )


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """
    Compute a stable hash of a configuration dictionary.

    Args:
        config_dict: Configuration to hash

    Returns:
        8-character hex string
    """
    config_json = json.dumps(config_dict, sort_keys=True)
    hash_obj = hashlib.sha256(config_json.encode())
    return hash_obj.hexdigest()[:8]


def generate_dataset_name(
    methods: List[str],
    config_dict: Dict[str, Any],
    run_index: int = 1,
    custom_name: Optional[str] = None,
) -> str:
    """
    Generate a dataset name following the naming convention.

    Args:
        methods: List of generator methods used
        config_dict: Configuration dictionary for hashing
        run_index: Run number (1-indexed)
        custom_name: Optional custom name (overrides default)

    Returns:
        Dataset name string

    Example:
        >>> generate_dataset_name(["ER", "PA"], {"n": 10, "rho": 0.3}, 1)
        'blanket_graphs_ER-PA_a1b2c3d4__v01'
    """
    if custom_name:
        return custom_name

    methods_tag = "-".join(methods)
    config_hash = compute_config_hash(config_dict)
    run_suffix = f"v{run_index:02d}"

    return f"blanket_graphs_{methods_tag}_{config_hash}__{run_suffix}"
