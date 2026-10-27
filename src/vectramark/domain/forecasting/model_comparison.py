"""Compare forecasting baselines using chronological walk-forward validation."""

import math
from dataclasses import dataclass

import pandas as pd

from vectramark.domain.forecasting.backtesting import (
    BacktestResult,
    walk_forward_backtest,
)
from vectramark.domain.forecasting.baseline import naive_forecast
from vectramark.domain.forecasting.validation import validate_closing_prices

MOVING_AVERAGE_WINDOW = 5


@dataclass(frozen=True)
class ModelComparisonResult:
    """Comparable backtest results for two forecasting baselines."""

    naive_result: BacktestResult
    moving_average_result: BacktestResult
    winner: str


def moving_average_forecast(
    closing_prices: pd.Series,
    horizon: int = 1,
    min_observations: int = 10,
    window: int = MOVING_AVERAGE_WINDOW,
) -> pd.Series:
    """Forecast using the mean of the latest observed closing prices."""

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer.")

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an integer.")

    if window < 1:
        raise ValueError("window must be at least 1.")

    prices = validate_closing_prices(
        closing_prices,
        min_observations=min_observations,
    )

    if window > len(prices):
        raise ValueError("window cannot exceed available historical observations.")

    average_price = float(prices.iloc[-window:].mean())

    return pd.Series(
        [average_price] * horizon,
        name="forecast",
        dtype=float,
    )


def compare_forecasters(
    closing_prices: pd.Series,
    initial_train_size: int,
    horizon: int = 5,
) -> ModelComparisonResult:
    """Compare both models on identical chronological test observations."""

    naive_result = walk_forward_backtest(
        closing_prices=closing_prices,
        initial_train_size=initial_train_size,
        horizon=horizon,
        forecaster=naive_forecast,
    )

    moving_average_result = walk_forward_backtest(
        closing_prices=closing_prices,
        initial_train_size=initial_train_size,
        horizon=horizon,
        forecaster=moving_average_forecast,
    )

    if not naive_result.actual.index.equals(moving_average_result.actual.index):
        raise ValueError("Models must be evaluated on identical timestamps.")

    naive_mae = naive_result.metrics["mae"]
    moving_average_mae = moving_average_result.metrics["mae"]

    if math.isclose(naive_mae, moving_average_mae, rel_tol=1e-12, abs_tol=1e-12):
        winner = "tie"
    elif naive_mae < moving_average_mae:
        winner = "naive_baseline"
    else:
        winner = "moving_average_baseline"

    return ModelComparisonResult(
        naive_result=naive_result,
        moving_average_result=moving_average_result,
        winner=winner,
    )
