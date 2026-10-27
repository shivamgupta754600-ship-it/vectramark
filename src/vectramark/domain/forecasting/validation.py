"""Validation utilities for financial forecasting."""

import numpy as np
import pandas as pd


def validate_closing_prices(
    closing_prices: pd.Series,
    min_observations: int = 10,
) -> pd.Series:
    """Validate and return a clean series of historical closing prices.

    Args:
        closing_prices: Historical closing prices in chronological order.
        min_observations: Minimum number of observations required.

    Returns:
        A validated copy of the closing-price series.

    Raises:
        TypeError: If the input is not a pandas Series.
        ValueError: If the data is invalid or insufficient.
    """
    if not isinstance(closing_prices, pd.Series):
        raise TypeError("Closing prices must be a pandas Series.")

    if isinstance(min_observations, bool) or not isinstance(min_observations, int):
        raise TypeError("min_observations must be an integer.")

    if min_observations < 2:
        raise ValueError("min_observations must be at least 2.")

    if closing_prices.empty:
        raise ValueError("Closing prices cannot be empty.")

    if len(closing_prices) < min_observations:
        raise ValueError(f"At least {min_observations} observations are required.")

    prices = pd.to_numeric(closing_prices.copy(), errors="raise")

    values = prices.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Closing prices must contain only finite values.")

    if (values <= 0).any():
        raise ValueError("Closing prices must be positive.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Closing prices must be in chronological order.")

    if prices.index.has_duplicates:
        raise ValueError("Closing prices cannot contain duplicate timestamps.")

    return prices.astype(float)
