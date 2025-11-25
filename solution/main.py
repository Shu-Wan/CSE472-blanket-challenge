import csv
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from blanket.metrics import jaccard_score, rmse
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.model_loading import ModelSource
from tqdm import tqdm

from finetuner import TabPFNFineTuner
from generator import TabPFNEmbeddingGenerator
from mbmlp import MBMLPModel
from util import (
    create_run_directory,
    get_run_filepath,
    save_run_info,
    update_results_file,
)


@dataclass
class RunArtifacts:
    results: dict
    finetuned_regressors: dict
    embedding_generators: dict
    mb_models: dict
    best_thresholds: dict
    run_dir: Path


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeyondTheBlanket:
    def __init__(self, config_path: str = None):
        """
        Initialize the BeyondTheBlanket pipeline.

        Args:
            config_path: Path to the config.yaml file. If None, uses default path.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get("seed", 42)
        self._set_global_seed(self.seed)

        # Store hyperparameters as instance variables
        self.supported_feature_dims = self.config["supported_feature_dims"]
        self.holdout_fraction = self.config["holdout_fraction"]

        # TabPFN parameters
        self.tabpfn_epochs = self.config["tabpfn"]["epochs"]
        self.tabpfn_lr = self.config["tabpfn"]["lr"]
        self.tabpfn_max_samples = self.config["tabpfn"]["max_samples"]
        self.tabpfn_val_fraction = self.config["tabpfn"]["val_fraction"]
        self.tabpfn_model_path = self.config["tabpfn"]["model_path"]

        # MB MLP parameters
        self.mb_epochs = self.config["mb_mlp"]["epochs"]
        self.mb_lr = self.config["mb_mlp"]["lr"]
        self.mb_batch_size = self.config["mb_mlp"]["batch_size"]
        self.mb_hidden_sizes = tuple(self.config["mb_mlp"]["hidden_sizes"])
        self.mb_val_fraction = self.config["mb_mlp"]["val_fraction"]
        self.mb_n_fold = self.config["mb_mlp"]["n_fold"]

        # Threshold tuning parameters
        self.threshold_min = self.config["threshold_tuning"]["min"]
        self.threshold_max = self.config["threshold_tuning"]["max"]
        self.threshold_num = self.config["threshold_tuning"]["num_thresholds"]

        # Load datasets
        self.develop = load_dataset(
            "CSE472-blanket-challenge/final-dataset", "develop", split="train"
        )
        self.submit = load_dataset(
            "CSE472-blanket-challenge/final-dataset", "submit", split="train"
        )

        # Download model
        regressor_models = ModelSource.get_regressor_v2_5()
        self.model_path = hf_hub_download(
            repo_id=regressor_models.repo_id,
            filename="tabpfn-v2.5-regressor-v2.5_default.ckpt",
            local_dir=self.tabpfn_model_path,
        )

    def evaluate_threshold(
        self,
        threshold: float,
        tasks: list,
        emb_gen: TabPFNEmbeddingGenerator,
        mb_model: MBMLPModel,
        regressor_batched: TabPFNRegressor,
    ):
        """
        Evaluate the threshold for the MB model.

        Args:
            threshold: float, threshold for the probability
            tasks: list, list of tasks
            emb_gen: TabPFNEmbeddingGenerator, the embedding generator
            mb_model: MBMLPModel, the MB model
            rmses = []
            jaccs: list[float], list of Jaccard scores
        """
        rmses = []
        jaccs = []

        for task in tasks:
            X_emb_test, idx_slices = emb_gen.build_mb_test_data([task])
            mb_probs_sample = mb_model.predict_proba(X_emb_test)

            (start, end) = idx_slices[0]
            mb_probs_ds = mb_probs_sample[start:end, :]
            mb_mean = mb_probs_ds.mean(axis=0)

            mb_pred = (mb_mean >= threshold).astype(int)
            mb_true = np.asarray(task["feature_mask"])

            jaccs.append(jaccard_score(mb_true, mb_pred))

            X_train_task = np.asarray(task["X_train"])
            y_train_task = np.asarray(task["y_train"])
            X_test_task = np.asarray(task["X_test"])
            y_test_task = np.asarray(task["y_test"])

            # Select MB features.
            X_train_mb = X_train_task[:, mb_pred == 1]
            X_test_mb = X_test_task[:, mb_pred == 1]

            # Edge case: no features selected at this threshold.
            if X_train_mb.shape[1] == 0:
                # Fall back to using all features.
                X_train_mb = X_train_task
                X_test_mb = X_test_task
                # Optionally, treat MB as all ones for regression only.
                # mb_pred = np.ones_like(mb_pred)

            eval_reg = clone_model_for_evaluation(
                regressor_batched, {}, TabPFNRegressor
            )
            eval_reg.fit(X_train_mb, y_train_task)
            y_pred = eval_reg.predict(X_test_mb)

            rmses.append(rmse(y_test_task, y_pred))

        avg_rmse = float(np.mean(rmses))
        avg_jacc = float(np.mean(jaccs))
        score = avg_rmse * (1 - avg_jacc)
        return avg_rmse, avg_jacc, score

    def make_predictions(self, submit: list, artifacts: RunArtifacts):
        logger.info("Making predictions on submit dataset.")
        submission_results = []
        skipped_tasks = []
        finetuned_regressors = artifacts.finetuned_regressors
        embedding_generators = artifacts.embedding_generators
        mb_models = artifacts.mb_models
        best_thresholds = artifacts.best_thresholds
        run_dir = artifacts.run_dir

        for task in tqdm(submit, desc="Processing Submit Tasks"):
            data_id = task["data_id"]

            X_train_submit = np.asarray(task["X_train"])
            y_train_submit = np.asarray(task["y_train"])
            X_test_submit = np.asarray(task["X_test"])
            n_features = X_train_submit.shape[1]

            logger.info(f"Processing task {data_id} with {n_features} features.")

            if n_features not in finetuned_regressors:
                logger.warning(
                    f"No model trained for n_features={n_features}, skipping task {data_id}."
                )
                skipped_tasks.append(data_id)
                continue

            regressor = finetuned_regressors[n_features]
            emb_gen = embedding_generators[n_features]
            mb_model = mb_models[n_features]

            # Predict Markov Blanket using the MB model
            X_emb_test, idx_slices = emb_gen.build_mb_test_data([task])
            mb_probs_sample = mb_model.predict_proba(X_emb_test)

            (start, end) = idx_slices[0]
            mb_probs_ds = mb_probs_sample[start:end, :]
            mb_mean = mb_probs_ds.mean(axis=0)

            # Use tuned threshold per feature dimension
            if n_features in best_thresholds:
                mb_threshold = best_thresholds[n_features]
            else:
                mb_threshold = 0.5

            mb_pred_mask = (mb_mean >= mb_threshold).astype(int)

            # Predict y using the fine tuned regressor with MB filtered features
            X_train_mb = X_train_submit[:, mb_pred_mask == 1]
            X_test_mb = X_test_submit[:, mb_pred_mask == 1]

            # Handle edge case where no features are selected
            if X_train_mb.shape[1] == 0:
                logger.warning(
                    f"No features selected for task {data_id}, using all features"
                )
                X_train_mb = X_train_submit
                X_test_mb = X_test_submit
                mb_pred_mask = np.ones(n_features, dtype=int)

            # Clone the regressor for evaluation and make predictions
            eval_reg = clone_model_for_evaluation(regressor, {}, TabPFNRegressor)
            eval_reg.fit(X_train_mb, y_train_submit)
            y_pred = eval_reg.predict(X_test_mb)

            # Store results
            submission_results.append(
                {
                    "data_id": data_id,
                    "y_pred": y_pred.tolist(),
                    "markov_blanket_pred": mb_pred_mask.tolist(),
                }
            )

        csv_path = get_run_filepath(run_dir, "submission.csv")
        logger.info(f"Saving results to {csv_path}")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["data_id", "y_pred", "markov_blanket_pred"])

            for result in submission_results:
                writer.writerow(
                    [
                        result["data_id"],
                        json.dumps(result["y_pred"]),
                        json.dumps(result["markov_blanket_pred"]),
                    ]
                )

        update_results_file(
            run_dir,
            {
                "submission_results": submission_results,
                "submission_summary": {
                    "total_tasks": len(submit),
                    "processed_tasks": len(submission_results),
                    "skipped_tasks": skipped_tasks,
                },
            },
        )

    def run_pipeline(self) -> RunArtifacts:
        run_dir = create_run_directory()
        logger.info(f"Created run directory: {run_dir}")

        save_run_info(run_dir, self.config)

        finetuned_regressors = {}
        embedding_generators = {}
        mb_models = {}
        best_thresholds = {}
        all_results = {}

        for feat_dim in self.supported_feature_dims:
            # Filter develop to this feature dimension and split train / heldout
            data_dim = [d for d in self.develop if d["n_features"] == feat_dim]
            if len(data_dim) == 0:
                logger.warning(
                    f"No tasks found for n_features == {feat_dim}, skipping."
                )
                continue

            logger.info(
                f"Processing feature dim {feat_dim}, total tasks: {len(data_dim)}"
            )

            n_total = len(data_dim)
            n_heldout = max(1, int(n_total * self.holdout_fraction))
            n_train_all = n_total - n_heldout

            train_tasks_all = data_dim[:n_train_all]
            heldout_tasks = data_dim[n_train_all:]

            logger.info(f"Train tasks (for finetune + MBMLP): {len(train_tasks_all)}")
            logger.info(f"Heldout tasks: {len(heldout_tasks)}")

            # Fine tune TabPFN on train_tasks_all for this dim with an internal validation fraction
            finetuner = TabPFNFineTuner(
                model_path=self.model_path,
                max_samples=self.tabpfn_max_samples,
                epochs=self.tabpfn_epochs,
                lr=self.tabpfn_lr,
            )

            # Set n_tasks to None to use all tasks and val_fraction for validation
            finetuner.prepare_from_develop(
                train_tasks_all, n_tasks=None, val_fraction=self.tabpfn_val_fraction
            )
            finetuner.fine_tune()

            regressor_batched = finetuner.get_regressor()
            finetuned_regressors[feat_dim] = regressor_batched

            # Clone in eval mode for embeddings and regression
            regressor_eval = clone_model_for_evaluation(
                regressor_batched,
                {},
                TabPFNRegressor,
            )

            # Build embeddings for train_tasks_all and train MB NN
            emb_gen = TabPFNEmbeddingGenerator(regressor_eval, n_fold=self.mb_n_fold)
            embedding_generators[feat_dim] = emb_gen

            mb_dim = feat_dim
            X_emb_train, y_mb_train = emb_gen.build_mb_training_data(
                tasks=train_tasks_all,
                expected_mb_dim=mb_dim,
            )

            emb_dim = X_emb_train.shape[1]
            mb_nn = MBMLPModel(
                input_dim=emb_dim,
                mb_dim=mb_dim,
                hidden_sizes=self.mb_hidden_sizes,
                lr=self.mb_lr,
                epochs=self.mb_epochs,
                batch_size=self.mb_batch_size,
            )

            mb_nn.fit(
                X_emb_train, y_mb_train, val_fraction=self.mb_val_fraction, shuffle=True
            )
            train_losses = mb_nn.get_losses()
            val_losses = mb_nn.get_val_losses()
            mb_models[feat_dim] = mb_nn

            # Plot MB NN training and validation loss curves, save to file
            plt.figure()
            plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
            plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"MBMLP train and val loss (n_features={feat_dim})")
            plt.grid(True)
            plt.legend()
            plot_path = get_run_filepath(run_dir, f"mbmlp_loss_dim{feat_dim}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

            # Tune MB threshold for this dim on heldout_tasks
            thresholds = np.linspace(
                self.threshold_min, self.threshold_max, self.threshold_num
            )

            best_t = None
            best_score = float("inf")

            logger.info(f"Tuning MB threshold for dim={feat_dim} on heldout tasks.")
            for t in thresholds:
                avg_rmse_t, avg_jacc_t, score_t = self.evaluate_threshold(
                    t, heldout_tasks, emb_gen, mb_nn, regressor_batched
                )

                if score_t < best_score:
                    best_score = score_t
                    best_t = t

            logger.info(
                f"Best threshold for dim={feat_dim}: {best_t} with score={best_score:.4f}"
            )
            best_thresholds[feat_dim] = best_t

            # Evaluate on all heldout tasks (MB + regression) for this dim using the tuned threshold
            dim_results = []

            logger.info(
                f"Evaluating on {len(heldout_tasks)} heldout tasks for dim={feat_dim}."
            )
            for task in heldout_tasks:
                X_emb_test, idx_slices = emb_gen.build_mb_test_data([task])
                mb_probs_sample = mb_nn.predict_proba(X_emb_test)

                (start, end) = idx_slices[0]
                mb_probs_ds = mb_probs_sample[start:end, :]
                mb_mean = mb_probs_ds.mean(axis=0)
                mb_pred_mask = (mb_mean >= best_t).astype(int)

                mb_true_mask = np.asarray(task["feature_mask"])
                jacc = float(jaccard_score(mb_true_mask, mb_pred_mask))

                # Regression with MB filtered features
                X_train_task = np.asarray(task["X_train"])
                y_train_task = np.asarray(task["y_train"])
                X_test_task = np.asarray(task["X_test"])
                y_test_task = np.asarray(task["y_test"])

                X_train_mb = X_train_task[:, mb_pred_mask == 1]
                X_test_mb = X_test_task[:, mb_pred_mask == 1]

                eval_reg = clone_model_for_evaluation(
                    regressor_batched, {}, TabPFNRegressor
                )
                eval_reg.fit(X_train_mb, y_train_task)
                y_pred = eval_reg.predict(X_test_mb)

                rmse_val = float(rmse(y_test_task, y_pred))

                dim_results.append(
                    {
                        "data_id": task["data_id"],
                        "rmse": rmse_val,
                        "jaccard": jacc,
                        "markov_blanket_pred": mb_pred_mask.astype(int).tolist(),
                        "y_pred": y_pred.tolist(),
                    }
                )

            all_rmse = float(np.mean([r["rmse"] for r in dim_results]))
            all_jacc = float(np.mean([r["jaccard"] for r in dim_results]))
            final_score = float(all_rmse * (1.0 - all_jacc))

            all_results[feat_dim] = {
                "per_task": dim_results,
                "avg_rmse": all_rmse,
                "avg_jaccard": all_jacc,
                "score": final_score,
            }

        per_task_rmse = []
        per_task_jacc = []
        for dim_metrics in all_results.values():
            for task_metrics in dim_metrics["per_task"]:
                per_task_rmse.append(task_metrics["rmse"])
                per_task_jacc.append(task_metrics["jaccard"])

        overall_summary = {}
        if per_task_rmse:
            avg_rmse = float(np.mean(per_task_rmse))
            avg_jaccard = float(np.mean(per_task_jacc))
            overall_summary = {
                "avg_rmse": avg_rmse,
                "avg_jaccard": avg_jaccard,
                "score": float(avg_rmse * (1.0 - avg_jaccard)),
            }

        update_results_file(
            run_dir,
            {
                "develop_metrics": all_results,
                "overall_develop_summary": overall_summary,
                "best_thresholds": best_thresholds,
            },
        )

        return RunArtifacts(
            results=all_results,
            finetuned_regressors=finetuned_regressors,
            embedding_generators=embedding_generators,
            mb_models=mb_models,
            best_thresholds=best_thresholds,
            run_dir=run_dir,
        )

    @staticmethod
    def _set_global_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    start_time = time.time()
    bt = BeyondTheBlanket()
    artifacts = bt.run_pipeline()
    bt.make_predictions(bt.submit, artifacts)
    total_runtime = time.time() - start_time
    update_results_file(
        artifacts.run_dir,
        {"overall_runtime_seconds": float(total_runtime)},
    )
