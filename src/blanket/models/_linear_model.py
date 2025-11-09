import warnings
from typing import Any

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define training function
def linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_mask: np.ndarray | None = None,
    seed: int = 42,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """
    End-to-end training and evaluation of a linear regression model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_nodes - 1)
    y : np.ndarray
        Target variable of shape (n_samples,)
    feature_mask : np.ndarray, optional
        Boolean mask for features to use (length = n_nodes - 1)
        If None, uses all features.
    seed : int, optional
        Random seed for train/test split
    **model_kwargs
        Additional arguments for the LinearRegression model

    Returns
    -------
    dict or None
        Dictionary with keys: 'model', 'scaler', 'rmse', 'mae', 'r2', 'n_features', 'predictions'
        Returns None if no features are selected
    """
    if feature_mask is None:
        feature_mask = np.ones(X.shape[1], dtype=bool)
    feature_mask = np.asarray(feature_mask, dtype=bool)
    assert np.sum(feature_mask) > 0, "No features selected in feature_mask."

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X[:, feature_mask], y, test_size=0.2, random_state=seed
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression(**model_kwargs)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

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
        "n_features": np.sum(feature_mask),
        "predictions": y_pred,
        "truths": y_test,
    }


def linear_l1_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_mask: np.ndarray | None = None,
    alpha: float | None = None,
    seed: int = 42,
    threshold: float = 1e-5,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """End-to-end training and evaluation of L1-regularized linear regression (Lasso).

    This function performs feature selection via L1 regularization and returns
    both the model and the selected features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target variable of shape (n_samples,).
    feature_mask : np.ndarray, optional
        Boolean mask for initial feature selection. If None, uses all features.
    alpha : float, optional
        Regularization strength. If None, uses LassoCV for cross-validation.
    seed : int, default=42
        Random seed for train/test split.
    threshold : float, default=1e-5
        Threshold for considering a coefficient as non-zero.
    **model_kwargs
        Additional arguments for the Lasso model.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': Final LinearRegression model trained on selected features
          (None if no features selected)
        - 'l1_model': Original Lasso model used for feature selection
        - 'scaler': StandardScaler used for feature scaling
        - 'rmse': Root mean squared error on test set
        - 'mae': Mean absolute error on test set
        - 'r2': R² score on test set
        - 'n_features': Number of features used (after initial mask)
        - 'n_selected': Number of features selected by L1
        - 'selected_features': Boolean mask of features selected by L1
        - 'predictions': Predictions on test set
        - 'truths': True values on test set
        - 'alpha': Alpha value used (from LassoCV if applicable)
    """
    # Apply initial feature mask if provided
    if feature_mask is None:
        feature_mask = np.ones(X.shape[1], dtype=bool)
    feature_mask = np.asarray(feature_mask, dtype=bool)
    assert np.sum(feature_mask) > 0, "No features selected in feature_mask."

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X[:, feature_mask], y, test_size=0.2, random_state=seed
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with L1 regularization
    if alpha is None:
        # Use cross-validation to find best alpha
        model = LassoCV(random_state=seed, **model_kwargs)
        model.fit(X_train_scaled, y_train)
        best_alpha = model.alpha_
    else:
        model = Lasso(alpha=alpha, random_state=seed, **model_kwargs)
        model.fit(X_train_scaled, y_train)
        best_alpha = alpha

    # Get selected features based on non-zero coefficients
    l1_selected = np.abs(model.coef_) > threshold

    # Retrain final model on only selected features
    # NOTE: The other option is to set non-selected feature coef to 0 to avoid re-training
    if np.sum(l1_selected) > 0:
        X_train_selected = X_train_scaled[:, l1_selected]
        X_test_selected = X_test_scaled[:, l1_selected]

        final_model = LinearRegression(**model_kwargs)
        final_model.fit(X_train_selected, y_train)
        y_pred = final_model.predict(X_test_selected)
    else:
        # No features selected, return zero predictions
        warnings.warn(
            "L1 regularization selected no features; returning zero predictions."
        )
        final_model = None
        y_pred = np.zeros_like(y_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Map L1 selection back to original feature space
    full_selected = np.zeros(X.shape[1], dtype=bool)
    full_selected[feature_mask] = l1_selected

    return {
        "model": final_model,
        "l1_model": model,  # Keep L1 model for reference
        "scaler": scaler,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_features": np.sum(l1_selected),  # feature used in prediction
        "selected_features": full_selected,
        "predictions": y_pred,
        "truths": y_test,
        "alpha": best_alpha,
    }
