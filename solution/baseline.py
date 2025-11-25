import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor
from tabpfn.model_loading import ModelSource

from blanket.metrics import jaccard_score, rmse

load_dotenv(override=True)


def evaluate_baseline(config_path: str):
    config = yaml.safe_load(Path(config_path).read_text())
    seed = config.get("seed", 42)
    _set_global_seed(seed)
    develop = load_dataset(
        "CSE472-blanket-challenge/final-dataset", "develop", split="train"
    )

    results = {}
    all_task_metrics = []  # Collect all task metrics across all dimensions
    model_ckpt_path = _resolve_model_checkpoint(config["tabpfn"]["model_path"])
    reg = TabPFNRegressor(
        model_path=model_ckpt_path,
        fit_mode="fit_preprocessors",
        differentiable_input=False,
        ignore_pretraining_limits=True,
        device="cuda",
    )

    for feat_dim in config["supported_feature_dims"]:
        tasks = [d for d in develop if d["n_features"] == feat_dim]
        if not tasks:
            continue

        n_total = len(tasks)
        n_heldout = max(1, int(n_total * config["holdout_fraction"]))
        heldout = tasks[n_total - n_heldout :]

        dim_metrics = []
        for task in heldout:
            X_tr = np.asarray(task["X_train"])
            y_tr = np.asarray(task["y_train"])
            X_te = np.asarray(task["X_test"])
            y_te = np.asarray(task["y_test"])

            reg.fit(X_tr, y_tr)
            y_pred = reg.predict(X_te)

            rmse_val = float(rmse(y_te, y_pred))
            blanket_pred = np.ones(
                task["n_features"], dtype=int
            )  # all features selected
            blanket_true = np.asarray(task["feature_mask"])
            jacc = float(jaccard_score(blanket_true, blanket_pred))
            combined = rmse_val * (1.0 - jacc)

            task_result = {"rmse": rmse_val, "jaccard": jacc, "combined": combined}
            dim_metrics.append(task_result)
            all_task_metrics.append(task_result)

        if dim_metrics:
            avg_rmse = float(np.mean([m["rmse"] for m in dim_metrics]))
            avg_jaccard = float(np.mean([m["jaccard"] for m in dim_metrics]))
            results[feat_dim] = {
                "avg_rmse": avg_rmse,
                "avg_jaccard": avg_jaccard,
                "score": float(avg_rmse * (1.0 - avg_jaccard)),
            }

    if all_task_metrics:
        overall_avg_rmse = float(np.mean([m["rmse"] for m in all_task_metrics]))
        overall_avg_jaccard = float(np.mean([m["jaccard"] for m in all_task_metrics]))
        results["overall"] = {
            "avg_rmse": overall_avg_rmse,
            "avg_jaccard": overall_avg_jaccard,
            "score": float(overall_avg_rmse * (1.0 - overall_avg_jaccard)),
        }

    return results


def _resolve_model_checkpoint(configured_path: str) -> str:
    """
    Ensure we have an actual TabPFN checkpoint file to load.

    If the configured path already points to a file, use it as-is. Otherwise,
    download the default v2.5 checkpoint into the provided directory.
    """
    path = Path(configured_path).expanduser()
    if path.is_file():
        return str(path)

    path.mkdir(parents=True, exist_ok=True)
    regressor_models = ModelSource.get_regressor_v2_5()
    ckpt_name = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
    logging.info("Downloading TabPFN checkpoint to %s", path)
    ckpt_path = hf_hub_download(
        repo_id=regressor_models.repo_id,
        filename=ckpt_name,
        local_dir=str(path),
    )
    return ckpt_path


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Evaluate TabPFN baseline metrics.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Evaluating baseline using config: %s", args.config)
    results = evaluate_baseline(args.config)

    if not results:
        logging.warning("No results produced. Check supported_feature_dims or dataset.")
        return

    for feat_dim, metrics in results.items():
        logging.info(
            "n_features=%s | avg_rmse=%.4f | avg_jaccard=%.4f | score=%.4f",
            feat_dim,
            metrics["avg_rmse"],
            metrics["avg_jaccard"],
            metrics["score"],
        )


if __name__ == "__main__":
    main()
