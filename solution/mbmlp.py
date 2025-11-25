from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MBMLP(nn.Module):
    """Simple feedforward network used to predict feature masks."""

    def __init__(
        self, input_dim: int, mb_dim: int, hidden_sizes: Sequence[int] = (256, 128)
    ) -> None:
        """
        Args:
            input_dim: Dimensionality of the embedding inputs.
            mb_dim: Dimensionality of the mask outputs.
            hidden_sizes: Sizes of hidden layers in the MLP stack.
        """
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, mb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.net(x)


class MBMLPModel:
    """Wrapper handling MBMLP training, validation, and inference."""

    def __init__(
        self,
        input_dim: int,
        mb_dim: int,
        lr: float,
        epochs: int,
        batch_size: int,
        hidden_sizes: Sequence[int] = (256, 128),
    ) -> None:
        """
        Args:
            input_dim: Dimensionality of embedding inputs.
            mb_dim: Dimensionality of the mask outputs.
            lr: Learning rate for Adam optimizer.
            epochs: Number of training epochs.
            batch_size: Batch size used for optimization.
            hidden_sizes: Layer widths for the MLP body.
        """
        self.input_dim = input_dim
        self.mb_dim = mb_dim
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cuda"

        self.model = MBMLP(input_dim, mb_dim, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

        # Store training and validation loss per epoch.
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def fit(
        self,
        X_emb: npt.ArrayLike,
        y_mb: npt.ArrayLike,
        val_fraction: float = 0.2,
        shuffle: bool = True,
    ) -> MBMLPModel:
        """
        Fit the MLP to embeddings and feature-mask labels.

        Args:
            X_emb: Embedding array with shape ``(n_samples, emb_dim)``.
            y_mb: Binary masks with shape ``(n_samples, mb_dim)``.
            val_fraction: Fraction of data held out for validation tracking.
            shuffle: Whether to shuffle before creating train/validation splits.

        Returns:
            Self reference to allow chaining.
        """
        X_emb = np.asarray(X_emb)
        y_mb = np.asarray(y_mb)

        n_samples = X_emb.shape[0]
        n_val = int(n_samples * val_fraction)
        n_train = n_samples - n_val

        if shuffle:
            idx = np.random.permutation(n_samples)
        else:
            idx = np.arange(n_samples)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:] if n_val > 0 else None

        X_train = X_emb[train_idx]
        y_train = y_mb[train_idx]

        if val_idx is not None:
            X_val = X_emb[val_idx]
            y_val = y_mb[val_idx]
        else:
            X_val = None
            y_val = None

        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        g = torch.Generator()
        g.manual_seed(42)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, generator=g
        )

        if X_val is not None:
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).float()
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            val_loader = None

        self.model.train()
        self.train_losses.clear()
        self.val_losses.clear()

        # Training loop.
        for epoch in range(self.epochs):
            epoch_train_loss = 0.0
            train_count = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item() * xb.size(0)
                train_count += xb.size(0)

            epoch_train_loss /= max(train_count, 1)
            self.train_losses.append(epoch_train_loss)

            # Validation loop.
            if val_loader is not None:
                self.model.eval()
                epoch_val_loss = 0.0
                val_count = 0

                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)

                        logits = self.model(xb)
                        loss = self.criterion(logits, yb)

                        epoch_val_loss += loss.item() * xb.size(0)
                        val_count += xb.size(0)

                epoch_val_loss /= max(val_count, 1)
                self.val_losses.append(epoch_val_loss)
                self.model.train()
            else:
                # No validation split.
                self.val_losses.append(float("nan"))

            # Log the training and validation losses.
            # logger.info(
            #     f"Epoch {epoch + 1}/{self.epochs}  "
            #     f"Train loss: {epoch_train_loss:.4f}  "
            #     f"Val loss: {epoch_val_loss:.4f}" if val_loader is not None else ""
            # )

        return self

    def get_losses(self) -> list[float]:
        """Return the history of averaged training losses."""
        return self.train_losses

    def get_val_losses(self) -> list[float]:
        """Return the history of averaged validation losses."""
        return self.val_losses

    def predict_proba(self, X_emb: npt.ArrayLike) -> np.ndarray:
        """
        Produce per-feature probabilities via sigmoid activation.

        Args:
            X_emb: Embeddings array whose rows are scored independently.

        Returns:
            Probability matrix with the same shape as ``y_mb``.
        """
        self.model.eval()
        X_tensor = torch.from_numpy(np.asarray(X_emb)).float().to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def predict(self, X_emb: npt.ArrayLike, threshold: float) -> np.ndarray:
        """
        Convert probabilities into binary mask decisions.

        Args:
            X_emb: Embedding matrix of shape ``(n_samples, emb_dim)``.
            threshold: Probability threshold applied element-wise.

        Returns:
            Integer mask array with entries of 0 or 1.
        """
        probs = self.predict_proba(X_emb)
        return (probs >= threshold).astype(int)
