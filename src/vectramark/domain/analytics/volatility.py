"""Historical volatility calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_historical_volatility(
    close: pd.Series,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate annualized rolling historical volatility.

    Volatility is expressed as a decimal return, not a percentage.
    Multiply the result by 100 to display it as a percentage.

    Example:
        0.20 represents 20% annualized historical volatility.
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")

    if close.empty:
        raise ValueError("close cannot be empty.")

    for name, value in (
        ("window", window),
        ("periods_per_year", periods_per_year),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if value < 1:
            raise ValueError(f"{name} must be greater than zero.")

    if close.isna().any():
        raise ValueError("close cannot contain missing values.")

    prices = pd.to_numeric(close, errors="raise").astype(float)

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("close must contain only finite values.")

    if (prices <= 0).any():
        raise ValueError("Closing prices must be greater than zero.")

    returns = prices.pct_change(fill_method=None)

    volatility = returns.rolling(window=window, min_periods=window).std(
        ddof=1
    ) * np.sqrt(periods_per_year)

    return volatility.rename(f"Historical_Volatility_{window}")
