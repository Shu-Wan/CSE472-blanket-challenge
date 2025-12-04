from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from tabpfn import TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding

TaskRecord = Mapping[str, Any]


class TabPFNEmbeddingGenerator:
    """Helper to derive TabPFN embeddings for meta-boost pipelines."""

    def __init__(self, regressor: TabPFNRegressor, n_fold: int = 5) -> None:
        """
        Args:
            regressor: Pre-trained TabPFN regressor used for embedding extraction.
            n_fold: Number of folds provided to ``TabPFNEmbedding``.
        """
        self.regressor = regressor
        self.n_fold = n_fold
        self.embedding_extractor = TabPFNEmbedding(
            tabpfn_reg=self.regressor,
            n_fold=self.n_fold,
        )

    def compute_task_embeddings(
        self,
        X_train: npt.ArrayLike,
        y_train: npt.ArrayLike,
        X_test: npt.ArrayLike,
        mode: str = "train",
    ) -> np.ndarray:
        """
        Compute embeddings for a single task using the configured regressor.

        Args:
            X_train: Training feature matrix.
            y_train: Training targets.
            X_test: Test feature matrix.
            mode: Data source identifier understood by ``TabPFNEmbedding``.

        Returns:
            ``numpy.ndarray`` of embeddings produced by the extractor.
        """
        emb = self.embedding_extractor.get_embeddings(
            np.asarray(X_train),
            np.asarray(y_train),
            np.asarray(X_test),
            data_source=mode,
        )
        return emb

    def build_mb_training_data(
        self, tasks: Sequence[TaskRecord], expected_mb_dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build averaged embeddings and tiled masks for training.

        Args:
            tasks: Sequence of task dictionaries with feature masks.
            expected_mb_dim: Expected dimensionality of each mask vector.

        Returns:
            Tuple ``(X_emb, y_mb)`` representing embeddings and mask labels.
        """
        train_embeddings = []
        mb_labels = []

        for d in tasks:
            X_tr = np.asarray(d["X_train"])
            y_tr = np.asarray(d["y_train"])
            X_te = np.asarray(d["X_test"])

            emb = self.compute_task_embeddings(X_tr, y_tr, X_te, mode="train")
            emb_agg = emb.mean(axis=0)
            train_embeddings.append(emb_agg)

            mb = np.asarray(d["feature_mask"])
            assert mb.shape[0] == expected_mb_dim

            n_train = int(d["n_train"])
            mb_tiled = np.tile(mb, (n_train, 1))
            mb_labels.append(mb_tiled)

        X_emb = np.concatenate(train_embeddings, axis=0)
        y_mb = np.concatenate(mb_labels, axis=0)
        return X_emb, y_mb

    def build_mb_test_data(
        self, tasks: Sequence[TaskRecord]
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """
        Build embeddings and index slices for evaluation.

        Args:
            tasks: Sequence of task dictionaries to embed.

        Returns:
            Tuple of ``(X_emb, idx_slices)`` where ``idx_slices`` maps each task
            back into the combined embedding array.
        """
        test_embeddings = []
        idx_slices = []
        offset = 0

        for d in tasks:
            X_tr = np.asarray(d["X_train"])
            y_tr = np.asarray(d["y_train"])
            X_te = np.asarray(d["X_test"])

            emb = self.compute_task_embeddings(X_tr, y_tr, X_te, mode="test")
            emb_agg = emb.mean(axis=0)

            n_samples = emb_agg.shape[0]
            start, end = offset, offset + n_samples
            idx_slices.append((start, end))
            offset = end

            test_embeddings.append(emb_agg)

        X_emb = np.concatenate(test_embeddings, axis=0)
        return X_emb, idx_slices
