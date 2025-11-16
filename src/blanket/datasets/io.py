"""Dataset I/O functions for Parquet + JSON storage."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset

from blanket.utils import read_json, save_json


def save_data(
    output_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """Save dataset

    Creates:
        output_path/<data_id>/train.parquet
        output_path/<data_id>/test.parquet
        output_path/<data_id>/meta.json

    Args:
        output_path: Base output path
        X_train: Training features (n_train, d)
        y_train: Training target (n_train,)
        X_test: Test features (n_test, d)
        y_test: Test target (n_test,)
        metadata: Metadata dictionary
    """
    data_id = metadata["data_id"]

    output_path = Path(output_path)
    data_path = output_path / data_id
    data_path.mkdir(parents=True, exist_ok=True)

    # Save train data
    train_df = pd.DataFrame({"X": list(X_train), "y": y_train})
    train_df.to_parquet(data_path / "train.parquet", index=False)

    # Save test data
    test_df = pd.DataFrame({"X": list(X_test), "y": y_test})
    test_df.to_parquet(data_path / "test.parquet", index=False)

    # Save metadata
    save_json(metadata, data_path / "meta.json")


def load_data(
    base_path: str | Path,
    data_id: str,
    include_metadata: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]
):
    """Load data from parquet files.

    Args:
        base_path: Base path where datasets are stored
        data_id: Unique identifier for the dataset
        include_metadata: Whether to include metadata dataset

    Returns:
        If include_metadata=False: (train_dataset, test_dataset) with
            X_train/y_train and X_test/y_test columns plus metadata
        If include_metadata=True: (train_dataset, test_dataset, metadata_dataset)
    """

    base_path = Path(base_path)
    data_path = base_path / data_id

    # Load train and test data
    train_ds = load_dataset(
        "parquet", data_files=str(data_path / "train.parquet"), split="train"
    )
    X_train = np.asarray(train_ds["X"])
    y_train = np.asarray(train_ds["y"])
    test_ds = load_dataset(
        "parquet", data_files=str(data_path / "test.parquet"), split="train"
    )
    X_test = np.asarray(test_ds["X"])
    y_test = np.asarray(test_ds["y"])

    if include_metadata:
        # Load metadata and convert to Dataset
        metadata_dict = read_json(data_path / "meta.json")
        return X_train, y_train, X_test, y_test, metadata_dict
    return X_train, y_train, X_test, y_test


def load_graph_data(dags_path: str | Path) -> Dataset:
    """Load graph dataset from file path or HuggingFace Hub.

    Args:
        dags_path: Path to graph JSONL file or HF Hub path (hf://repo/name)

    Returns:
        HuggingFace Dataset

    Raises:
        ValueError: If path format is invalid
    """
    dags_path_str = str(dags_path)

    ds = None
    if dags_path_str.startswith("hf://"):
        # Parse hf://repo/name format
        hf_path = dags_path_str[5:]  # Remove 'hf://' prefix
        ds = load_dataset(hf_path, split="train")
    elif dags_path_str.endswith(".jsonl") or dags_path_str.endswith(".json"):
        ds = load_dataset("json", data_files=dags_path_str, split="train")
    else:
        raise ValueError(
            f"Invalid dags_path format: {dags_path_str}. "
            "Must be 'hf://repo/name' or path to .jsonl/.json file."
        )

    return ds
