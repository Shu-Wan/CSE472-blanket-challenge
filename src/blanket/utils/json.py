"""
Helper functions for json.
"""

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _make_serializable(obj):
    """
    Recursively convert objects into JSON-serializable forms.

    This helper function makes objects JSON-serializable without converting to string.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)

    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]

    if isinstance(obj, (set, frozenset)):
        return sorted((_make_serializable(x) for x in obj), key=str)

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def serialize(obj: Any) -> str:
    """
    Recursively convert objects into JSON-serializable forms and return a stable JSON string.

    Returns a stable, order-insensitive JSON string with:
    - Keys sorted alphabetically
    - Compact separators (",", ":")
    - Non-ASCII preserved
    """
    serializable_obj = _make_serializable(obj)
    return json.dumps(
        serializable_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def read_json(file_path: str | Path):
    """
    Reads data from a JSON or JSONL file.

    Args:
        file_path (str): The path to the file.

    Returns:
        A dictionary or a list, depending on the file content.
        Returns a list of dictionaries for JSONL files.
    """
    file_path = str(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".jsonl"):
                # Read each line as a separate JSON object
                return [json.loads(line) for line in f]
            if file_path.endswith(".json"):
                # Read the entire file as a single JSON object/array
                return json.load(f)
            raise ValueError("Unsupported file format. Use .json or .jsonl")
    except FileNotFoundError as e:
        log.error("The file '%s' was not found: %s", file_path, e)
        return None
    except json.JSONDecodeError as e:
        log.error("Could not decode JSON from '%s': %s", file_path, e)
        return None
    except ValueError as e:
        log.error("Unsupported file format for '%s': %s", file_path, e)
        return None


def save_json(data, file_path: str | Path, indent: int = 2):
    """
    Saves data to a JSON or JSONL file.

    Args:
        data (dict or list): Data to save. If saving as .jsonl,
                             must be a list of dicts.
        file_path (str): The path to the output file.
        indent (int, optional): Indentation for JSON formatting.
                                Only applies to .json, not .jsonl.
    """
    file_path = str(file_path)

    try:
        if file_path.endswith(".jsonl"):
            # Must be iterable of records
            if not isinstance(data, (list, tuple)):
                raise ValueError(
                    "For .jsonl, data must be a list/tuple of dictionaries."
                )
            with open(file_path, "w", encoding="utf-8") as f:
                for obj in data:
                    safe = _make_serializable(obj)
                    f.write(json.dumps(safe, ensure_ascii=False) + "\n")

        elif file_path.endswith(".json"):
            safe = _make_serializable(data)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(safe, f, indent=indent, ensure_ascii=False)

        else:
            raise ValueError("Unsupported file format. Use .json or .jsonl")

    except OSError as e:
        log.error("Error writing '%s': %s", file_path, e)
    except (TypeError, ValueError) as e:
        log.error("Error writing '%s': %s", file_path, e)
