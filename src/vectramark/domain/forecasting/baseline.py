"""Baseline forecasting model for financial time series."""

import pandas as pd

from vectramark.domain.forecasting.validation import validate_closing_prices


def naive_forecast(
    closing_prices: pd.Series,
    horizon: int = 1,
    min_observations: int = 10,
) -> pd.Series:
    """Forecast future prices using the last observed closing price.

    Every forecast equals the most recent observed closing price.

    Args:
        closing_prices: Historical closing prices in chronological order.
        horizon: Number of future steps to forecast.
        min_observations: Minimum historical observations required.

    Returns:
        A series containing the forecast values.

    Raises:
        TypeError: If horizon is not an integer.
        ValueError: If horizon or historical data is invalid.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer.")

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    prices = validate_closing_prices(
        closing_prices,
        min_observations=min_observations,
    )

    last_price = float(prices.iloc[-1])

    forecast_values = [last_price] * horizon

    return pd.Series(
        forecast_values,
        name="forecast",
        dtype=float,
    )
