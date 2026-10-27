"""Historical return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_close(close: pd.Series) -> pd.Series:
    """Validate and return a numeric closing-price series."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")

    if close.empty:
        raise ValueError("close cannot be empty.")

    if close.isna().any():
        raise ValueError("close cannot contain missing values.")

    prices = pd.to_numeric(close, errors="raise").astype(float)

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("close must contain only finite values.")

    if (prices <= 0).any():
        raise ValueError("Closing prices must be greater than zero.")

    return prices


def calculate_daily_returns(close: pd.Series) -> pd.Series:
    """Calculate simple percentage returns between observations."""
    prices = _validate_close(close)

    return prices.pct_change(fill_method=None).rename("Daily_Return")


def calculate_cumulative_returns(close: pd.Series) -> pd.Series:
    """Calculate cumulative returns relative to the first observation."""
    prices = _validate_close(close)

    returns = prices.pct_change(fill_method=None)

    cumulative = (1 + returns).cumprod() - 1

    # The first observation is the starting point, so its return is zero.
    cumulative.iloc[0] = 0.0

    return cumulative.rename("Cumulative_Return")
