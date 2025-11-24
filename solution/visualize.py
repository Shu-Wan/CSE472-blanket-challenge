#!/usr/bin/env python3
"""
Inspect a single heldout task from a run’s results.json for more detailed diagnostics.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from blanket.plots import plot_graph
from datasets import load_dataset


def load_task_by_id(split, data_id):
    for task in split:
        if task["data_id"] == data_id:
            return task
    raise ValueError(f"Task {data_id} not found in develop split.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", required=True, help="Path to runs/<timestamp> directory"
    )
    parser.add_argument(
        "--data-id", required=True, help="Heldout task identifier to inspect"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config for dataset parameters"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store generated figures (defaults to <run>/visualizations)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text())

    # locate task metrics from results.json
    task_metrics = None
    for feat_dim, dim_data in results["develop_metrics"].items():
        for entry in dim_data["per_task"]:
            if entry["data_id"] == args.data_id:
                task_metrics = entry
                break
        if task_metrics:
            break
    if task_metrics is None:
        raise ValueError(f"{args.data_id} not in develop_metrics for run {args.run}")

    develop = load_dataset(
        "CSE472-blanket-challenge/final-dataset", "develop", split="train"
    )
    task = load_task_by_id(develop, args.data_id)

    # convert to numpy
    X_train = np.asarray(task["X_train"])
    X_test = np.asarray(task["X_test"])
    y_test = np.asarray(task["y_test"])
    blanket_true = np.asarray(task["feature_mask"])

    print(f"=== Task {args.data_id} ===")
    print(
        f"n_features: {task['n_features']} | train samples: {len(X_train)} | test samples: {len(X_test)}"
    )
    print(
        f"Recorded metrics: RMSE={task_metrics['rmse']:.4f}, Jaccard={task_metrics['jaccard']:.4f}, Combined={task_metrics['combined_score']:.4f}"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Causal graph visualization
    adj_matrix = np.asarray(task.get("adjacency_matrix"))
    if adj_matrix.size > 0:
        plot_graph(
            adj_matrix,
            title=f"Causal Graph: {task.get('graph_id', 'unknown')} ({args.data_id})",
            figsize=(8, 8),
        )
        graph_path = output_dir / f"{args.data_id}_graph.png"
        plt.savefig(graph_path, bbox_inches="tight")
        plt.close()
        print(f"Saved causal graph to {graph_path}")

    else:
        print("No causal graph found for this task.")

    # Blanket comparison (if stored in results)
    blanket_pred = np.asarray(task_metrics.get("markov_blanket_pred", []))
    if blanket_pred.size == blanket_true.size:
        plt.figure(figsize=(10, 3))
        idx = np.arange(len(blanket_true))
        plt.step(idx, blanket_true, where="mid", label="True blanket")
        plt.step(idx, blanket_pred, where="mid", label="Pred blanket")
        plt.ylim(-0.1, 1.1)
        plt.title(f"Markov Blanket vs Prediction ({args.data_id})")
        plt.xlabel("Feature index")
        plt.ylabel("Mask")
        plt.legend()
        plt.tight_layout()
        blanket_path = output_dir / f"{args.data_id}_blanket.png"
        plt.savefig(blanket_path, bbox_inches="tight")
        plt.close()
        print(f"Saved blanket comparison to {blanket_path}")
    else:
        print("No stored blanket prediction for this task.")

    # Residual plot using stored y_pred
    y_pred = np.asarray(task_metrics.get("y_pred", []))
    if y_pred.size == y_test.size and y_pred.size > 0:
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 4))
        plt.plot(residuals, marker="o")
        plt.axhline(0, color="k", linestyle="--")
        plt.title(f"Residuals on heldout test ({args.data_id})")
        plt.xlabel("Test sample index")
        plt.ylabel("y_true - y_pred")
        plt.tight_layout()
        residual_path = output_dir / f"{args.data_id}_residuals.png"
        plt.savefig(residual_path, bbox_inches="tight")
        plt.close()
        print(f"Saved residual plot to {residual_path}")
    else:
        print("No stored y_pred for this task; skipping residual plot.")


if __name__ == "__main__":
    main()
