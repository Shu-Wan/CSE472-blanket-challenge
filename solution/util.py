import json
from datetime import datetime
from pathlib import Path


def create_run_directory(root_dir: str = "runs") -> Path:
    """
    Create a timestamped run directory under the root directory.

    Args:
        root_dir: Root directory name for all runs (default: "runs")

    Returns:
        Path object for the created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_run_filepath(run_dir: Path, filename: str) -> Path:
    """
    Get the full path for a file within the run directory.

    Args:
        run_dir: The run directory Path object
        filename: Name of the file

    Returns:
        Path object for the file
    """
    return run_dir / filename


def save_run_info(run_dir: Path, config: dict):
    """
    Save run configuration info to the run directory.

    Args:
        run_dir: The run directory Path object
        config: Configuration dictionary to save
    """
    info_path = run_dir / "run_info.json"
    with open(info_path, "w") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "config": config}, f, indent=2
        )


def update_results_file(
    run_dir: Path, update: dict, filename: str = "results.json"
) -> Path:
    """
    Merge additional result data into results.json under the run directory.

    Args:
        run_dir: Directory for the current run.
        update: Dictionary of values to merge into the JSON.
        filename: Results file name relative to run_dir.

    Returns:
        Path to the results file.
    """
    results_path = run_dir / filename

    if results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
    else:
        data = {"timestamp": datetime.now().isoformat()}

    for key, value in update.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    return results_path
