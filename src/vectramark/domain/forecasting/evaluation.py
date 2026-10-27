"""Evaluation metrics for financial forecasting."""

import numpy as np
import pandas as pd


def evaluate_forecast(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and MAPE for forecast predictions.

    Args:
        actual: Actual observed prices.
        predicted: Corresponding forecast prices.

    Returns:
        A dictionary containing MAE, RMSE, and MAPE.

    Raises:
        TypeError: If either input is not a pandas Series.
        ValueError: If inputs are empty, invalid, unequal in length,
            or contain non-positive actual prices.
    """
    if not isinstance(actual, pd.Series) or not isinstance(predicted, pd.Series):
        raise TypeError("Actual and predicted values must be pandas Series.")

    if actual.empty or predicted.empty:
        raise ValueError("Actual and predicted values cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have equal lengths.")

    actual_values = pd.to_numeric(actual, errors="raise").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="raise").to_numpy(dtype=float)

    if not np.isfinite(actual_values).all():
        raise ValueError("Actual values must be finite.")

    if not np.isfinite(predicted_values).all():
        raise ValueError("Predicted values must be finite.")

    if (actual_values <= 0).any():
        raise ValueError("Actual prices must be positive for MAPE evaluation.")

    errors = actual_values - predicted_values

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mape = float(np.mean(np.abs(errors / actual_values)) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
