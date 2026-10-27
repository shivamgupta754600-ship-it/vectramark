"""Walk-forward backtesting utilities for financial forecasts."""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from vectramark.domain.forecasting.baseline import naive_forecast
from vectramark.domain.forecasting.evaluation import evaluate_forecast
from vectramark.domain.forecasting.validation import validate_closing_prices


@dataclass(frozen=True)
class BacktestResult:
    """Results from a chronological forecasting backtest."""

    actual: pd.Series
    predicted: pd.Series
    metrics: dict[str, float]
    initial_train_size: int
    horizon: int


def chronological_split(
    closing_prices: pd.Series,
    test_size: int,
    min_observations: int = 10,
) -> tuple[pd.Series, pd.Series]:
    """Split observations into chronological training and testing sets."""

    if isinstance(test_size, bool) or not isinstance(test_size, int):
        raise TypeError("test_size must be an integer.")

    if test_size < 1:
        raise ValueError("test_size must be at least 1.")

    prices = validate_closing_prices(
        closing_prices,
        min_observations=min_observations,
    )

    if test_size >= len(prices):
        raise ValueError("test_size must be smaller than the dataset length.")

    split_index = len(prices) - test_size

    train = prices.iloc[:split_index].copy()
    test = prices.iloc[split_index:].copy()

    return train, test


def walk_forward_backtest(
    closing_prices: pd.Series,
    initial_train_size: int,
    horizon: int = 1,
    min_observations: int = 10,
    forecaster: Callable[..., pd.Series] = naive_forecast,
) -> BacktestResult:
    """Evaluate forecasts using expanding-window walk-forward validation.

    At each evaluation step, the forecaster receives only observations
    available before the forecasted period. After each forecast, actual
    observations become available for subsequent forecasts.

    Args:
        closing_prices: Historical closing prices in chronological order.
        initial_train_size: Number of observations in the initial training set.
        horizon: Number of steps predicted at each forecast origin.
        min_observations: Minimum historical observations required.
        forecaster: Forecasting function accepting a price series and horizon.

    Returns:
        A BacktestResult containing actual values, predictions, and metrics.
    """

    if isinstance(initial_train_size, bool) or not isinstance(initial_train_size, int):
        raise TypeError("initial_train_size must be an integer.")

    if initial_train_size < min_observations:
        raise ValueError("initial_train_size must be at least min_observations.")

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer.")

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    if not callable(forecaster):
        raise TypeError("forecaster must be callable.")

    prices = validate_closing_prices(
        closing_prices,
        min_observations=min_observations,
    )

    if initial_train_size >= len(prices):
        raise ValueError("initial_train_size must be smaller than the dataset length.")

    actual_values: list[float] = []
    predicted_values: list[float] = []
    timestamps: list[object] = []

    forecast_start = initial_train_size

    while forecast_start < len(prices):
        forecast_end = min(forecast_start + horizon, len(prices))

        history = prices.iloc[:forecast_start]
        actual_chunk = prices.iloc[forecast_start:forecast_end]

        predictions = forecaster(history, horizon=len(actual_chunk))

        if not isinstance(predictions, pd.Series):
            raise TypeError("Forecaster must return a pandas Series.")

        if len(predictions) != len(actual_chunk):
            raise ValueError(
                "Forecaster must return exactly the requested number of predictions."
            )

        prediction_values = pd.to_numeric(
            predictions,
            errors="raise",
        ).to_numpy(dtype=float)

        if (
            not pd.Series(prediction_values)
            .map(lambda value: pd.notna(value) and abs(value) != float("inf"))
            .all()
        ):
            raise ValueError("Forecaster returned non-finite predictions.")

        if (prediction_values <= 0).any():
            raise ValueError("Forecast prices must be positive.")

        actual_values.extend(actual_chunk.astype(float).tolist())
        predicted_values.extend(prediction_values.tolist())
        timestamps.extend(actual_chunk.index.tolist())

        forecast_start = forecast_end

    actual_series = pd.Series(
        actual_values,
        index=pd.Index(timestamps),
        name="actual",
        dtype=float,
    )

    predicted_series = pd.Series(
        predicted_values,
        index=pd.Index(timestamps),
        name="predicted",
        dtype=float,
    )

    metrics = evaluate_forecast(actual_series, predicted_series)

    return BacktestResult(
        actual=actual_series,
        predicted=predicted_series,
        metrics=metrics,
        initial_train_size=initial_train_size,
        horizon=horizon,
    )
