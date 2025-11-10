"""Multi-layer perceptron regression models."""

from typing import Any

import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def mlp_regression(
    train: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    feature_mask: np.ndarray | None = None,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    **model_kwargs: Any,
) -> dict[str, Any]:
    """End-to-end training and evaluation of MLP regression model.

    Parameters
    ----------
    train : tuple[np.ndarray, np.ndarray]
        Training dataset as (X_train, y_train).
    test : tuple[np.ndarray, np.ndarray]
        Test dataset as (X_test, y_test).
    feature_mask : np.ndarray
        Boolean mask for features to use.
    seed : int, default=42
        Random seed for model.
    hidden_layer_sizes : tuple[int, ...], default=(100,)
        Number of neurons in each hidden layer.
    **model_kwargs
        Additional arguments for MLPRegressor.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': Fitted MLPRegressor
        - 'scaler': StandardScaler used for feature scaling
        - 'rmse': Root mean squared error on test set
        - 'mae': Mean absolute error on test set
        - 'r2': R² score on test set
        - 'n_features': Number of features used
        - 'predictions': Predictions on test set
        - 'truths': True values on test set
    """
    X_train, y_train = train
    X_test, y_test = test

    if feature_mask is None:
        feature_mask = np.ones(X_train.shape[1], dtype=bool)
    feature_mask = np.asarray(feature_mask, dtype=bool)
    assert np.sum(feature_mask) > 0, "No features selected in feature_mask."

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply feature mask
    X_train_masked = X_train_scaled[:, feature_mask]
    X_test_masked = X_test_scaled[:, feature_mask]

    # Train model
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, random_state=seed, **model_kwargs
    )
    model.fit(X_train_masked, y_train)
    y_pred = model.predict(X_test_masked)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_features": np.sum(feature_mask),  # num of features used for prediction
        "feature_mask": feature_mask,
        "predictions": y_pred,
        "truths": y_test,
    }


def mlp_l1_regression(
    train: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    alpha: float | None = None,
    seed: int = 42,
    threshold: float = 1e-5,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    **model_kwargs: Any,
) -> dict[str, Any]:
    """MLP regression with L1-based feature selection.

    This function first performs L1 feature selection using Lasso, then trains
    an MLP regression model on the selected features.

    Parameters
    ----------
    train : tuple[np.ndarray, np.ndarray]
        Training dataset as (X_train, y_train).
    test : tuple[np.ndarray, np.ndarray]
        Test dataset as (X_test, y_test).
    alpha : float, optional
        L1 regularization strength for feature selection. If None, uses LassoCV.
    seed : int, default=42
        Random seed for models.
    threshold : float, default=1e-5
        Threshold for considering a Lasso coefficient as non-zero.
    hidden_layer_sizes : tuple[int, ...], default=(100,)
        Number of neurons in each hidden layer of MLP.
    **model_kwargs
        Additional arguments for MLPRegressor.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': Fitted MLPRegressor
        - 'scaler': StandardScaler used for feature scaling
        - 'lasso_model': Lasso model used for feature selection
        - 'rmse': Root mean squared error on test set
        - 'mae': Mean absolute error on test set
        - 'r2': R² score on test set
        - 'n_features': Number of features after initial mask
        - 'n_selected': Number of features selected by L1
        - 'selected_features': Boolean mask of features selected by L1
        - 'predictions': Predictions on test set
        - 'truths': True values on test set
        - 'alpha': Alpha value used for L1 selection
    """
    X_train, y_train = train
    X_test, y_test = test

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 1: L1 feature selection using Lasso
    if alpha is None:
        lasso = LassoCV(random_state=seed)
        lasso.fit(X_train_scaled, y_train)
        best_alpha = lasso.alpha_
    else:
        lasso = Lasso(alpha=alpha, random_state=seed)
        lasso.fit(X_train_scaled, y_train)
        best_alpha = alpha

    # Get selected features based on non-zero coefficients
    l1_selected = np.abs(lasso.coef_) > threshold

    if np.sum(l1_selected) == 0:
        suggested_alpha = best_alpha / 10
        raise ValueError(
            f"L1 regularization selected no features with alpha={best_alpha:.4f}. "
            f"Try reducing alpha (e.g., {suggested_alpha:.4f}) or lowering the threshold "
            f"(current: {threshold})."
        )

    # Step 2: Train MLP on selected features
    X_train_selected = X_train_scaled[:, l1_selected]
    X_test_selected = X_test_scaled[:, l1_selected]

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, random_state=seed, **model_kwargs
    )
    mlp.fit(X_train_selected, y_train)
    y_pred = mlp.predict(X_test_selected)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": mlp,
        "scaler": scaler,
        "lasso_model": lasso,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_features": np.sum(l1_selected),  # num of features used for prediction
        "feature_mask": l1_selected,
        "predictions": y_pred,
        "truths": y_test,
        "alpha": best_alpha,
    }
