from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from blanket.metrics import rmse

TaskRecord = Mapping[str, Any]


class TabPFNFineTuner:
    """Fine-tunes a TabPFN regressor on a collection of meta-learning tasks."""

    def __init__(
        self,
        model_path: str,
        max_samples: int,
        epochs: int,
        lr: float,
    ) -> None:
        """
        Initialize the fine-tuner with a pre-trained TabPFN checkpoint.

        Args:
            model_path: Path to the TabPFN checkpoint directory.
            max_samples: Maximum samples to draw per task during preprocessing.
            epochs: Number of fine-tuning epochs.
            lr: Learning rate used by the Adam optimizer.
        """
        self.model_path = model_path
        self.max_samples = max_samples
        self.epochs = epochs
        self.model_config = {
            "ignore_pretraining_limits": True,
            "device": "cuda",
            "n_estimators": 24,
            "random_state": 42,
            "inference_precision": "auto",
        }

        self.regressor = TabPFNRegressor(
            model_path=self.model_path,
            fit_mode="batched",
            differentiable_input=False,
            **self.model_config,
        )

        self.regressor._initialize_model_variables()
        self.optimizer = Adam(self.regressor.model_.parameters(), lr=lr)
        self.datasets: Dataset | None = None
        self.loader: DataLoader | None = None
        self.val_tasks: list[TaskRecord] = []
        self.epoch_losses: list[float] = []
        self.val_rmse_history: list[float] = []

    def prepare_from_develop(
        self,
        develop: Sequence[TaskRecord],
        n_tasks: int | None = None,
        val_fraction: float = 0.0,
    ) -> None:
        """
        Build the PyTorch dataset and optionally split out validation tasks.

        Args:
            develop: Sequence of task dictionaries produced by the develop split.
            n_tasks: Optional cap on the number of tasks pulled from ``develop``.
            val_fraction: Fraction of selected tasks reserved for validation RMSE
                computation; ignored if fewer than two tasks remain.
        """
        if n_tasks is None:
            selected = develop
        else:
            n_tasks = min(n_tasks, len(develop))
            selected = (
                develop.select(range(n_tasks))
                if hasattr(develop, "select")
                else develop[:n_tasks]
            )

        selected_list = list(selected)

        if val_fraction > 0 and len(selected_list) > 1:
            n_total = len(selected_list)
            n_val = max(1, int(n_total * val_fraction))
            n_train = n_total - n_val
            train_list = selected_list[:n_train]
            val_list = selected_list[n_train:]
        else:
            train_list = selected_list
            val_list = []

        self.val_tasks = val_list

        train_test_split_seeded = partial(train_test_split, random_state=42)

        meta_datasets = []
        for d in train_list:
            X_train = np.asarray(d["X_train"])
            y_train = np.asarray(d["y_train"])

            ds = self.regressor.get_preprocessed_datasets(
                X_train,
                y_train,
                train_test_split_seeded,
                self.max_samples,
            )
            meta_datasets.append(ds)

        if len(meta_datasets) == 1:
            self.datasets = meta_datasets[0]
        else:
            self.datasets = ConcatDataset(meta_datasets)

        g = torch.Generator()
        g.manual_seed(42)

        self.loader = DataLoader(
            self.datasets,
            batch_size=1,
            shuffle=True,
            collate_fn=meta_dataset_collator,
            generator=g,
        )

    def fine_tune(self) -> None:
        """
        Run the fine-tuning loop and keep the checkpoint with the best RMSE.

        Raises:
            RuntimeError: If :meth:`prepare_from_develop` has not been executed.
        """
        if self.loader is None:
            raise RuntimeError(
                "DataLoader is not prepared. Call prepare_from_develop first."
            )

        self.epoch_losses.clear()
        self.val_rmse_history.clear()

        best_val_rmse = float("inf")
        best_state_dict = None

        for epoch in tqdm(range(self.epochs), desc="Fine tuning epochs", leave=False):
            total_loss = 0.0
            total_count = 0

            for data_batch in tqdm(self.loader, desc=f"Epoch {epoch}", leave=False):
                self.optimizer.zero_grad()

                (
                    X_tr,
                    X_te,
                    y_tr,
                    y_te,
                    cat_ixs,
                    confs,
                    raw_space,
                    znorm_space,
                    _,
                    _,
                ) = data_batch

                self.regressor.raw_space_bardist_ = raw_space[0]
                self.regressor.znorm_space_bardist_ = znorm_space[0]

                self.regressor.fit_from_preprocessed(X_tr, y_tr, cat_ixs, confs)

                preds, _, _ = self.regressor.forward(X_te)

                loss_fn = znorm_space[0]
                loss = loss_fn(preds, y_te.to(self.regressor.device)).mean()
                loss.backward()
                self.optimizer.step()

                batch_size = y_te.shape[0]
                total_loss += loss.item() * batch_size
                total_count += batch_size

            epoch_loss = total_loss / max(total_count, 1)
            self.epoch_losses.append(epoch_loss)

            if len(self.val_tasks) > 0:
                val_rmse = self._compute_val_rmse()
                self.val_rmse_history.append(val_rmse)
                print(
                    f"Epoch {epoch} - train loss: {epoch_loss:.4f} - val RMSE: {val_rmse:.4f}"
                )

                # Keep the best checkpoint.
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state_dict = deepcopy(self.regressor.model_.state_dict())
            else:
                self.val_rmse_history.append(float("nan"))
                print(f"Epoch {epoch} - train loss: {epoch_loss:.4f}")

        # After all epochs, restore the best weights.
        if best_state_dict is not None:
            self.regressor.model_.load_state_dict(best_state_dict)

    def _compute_val_rmse(self) -> float:
        """
        Compute RMSE metrics over the reserved validation tasks.

        Returns:
            Mean RMSE value across validation tasks, or ``nan`` if unavailable.
        """
        if not self.val_tasks:
            return float("nan")

        reg_eval = clone_model_for_evaluation(self.regressor, {}, TabPFNRegressor)
        rmses = []

        for d in self.val_tasks:
            X_tr = np.asarray(d["X_train"])
            y_tr = np.asarray(d["y_train"])
            X_te = np.asarray(d["X_test"])
            y_te = np.asarray(d["y_test"])

            reg_eval.fit(X_tr, y_tr)
            y_pred = reg_eval.predict(X_te)
            rmses.append(rmse(y_te, y_pred))

        return float(np.mean(rmses))

    def get_regressor(self) -> TabPFNRegressor:
        """
        Expose the underlying TabPFN regressor instance.

        Returns:
            The fine-tuned :class:`TabPFNRegressor`.
        """
        return self.regressor
