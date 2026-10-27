"""Tests for chronological splitting and walk-forward backtesting."""

import numpy as np
import pandas as pd
import pytest

from vectramark.domain.forecasting.backtesting import (
    chronological_split,
    walk_forward_backtest,
)


def sample_prices(count: int = 30) -> pd.Series:
    """Create deterministic chronological closing prices for tests."""

    return pd.Series(
        np.arange(100, 100 + count, dtype=float),
        index=pd.date_range("2026-01-01", periods=count, freq="D"),
        name="Close",
    )


def test_chronological_split_preserves_time_order() -> None:
    prices = sample_prices()

    train, test = chronological_split(prices, test_size=5)

    assert len(train) == 25
    assert len(test) == 5
    assert train.index[-1] < test.index[0]
    assert train.iloc[-1] == 124.0
    assert test.iloc[0] == 125.0


def test_chronological_split_rejects_invalid_test_size() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        chronological_split(sample_prices(), test_size=0)


def test_chronological_split_rejects_test_size_larger_than_data() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        chronological_split(sample_prices(10), test_size=10)


def test_walk_forward_backtest_returns_all_test_predictions() -> None:
    prices = sample_prices()

    result = walk_forward_backtest(
        prices,
        initial_train_size=20,
        horizon=3,
    )

    assert len(result.actual) == 10
    assert len(result.predicted) == 10
    assert result.initial_train_size == 20
    assert result.horizon == 3
    assert set(result.metrics) == {"mae", "rmse", "mape"}


def test_naive_forecast_uses_latest_available_price() -> None:
    prices = sample_prices(20)

    result = walk_forward_backtest(
        prices,
        initial_train_size=15,
        horizon=1,
    )

    assert result.predicted.iloc[0] == prices.iloc[14]
    assert result.predicted.iloc[1] == prices.iloc[15]


def test_walk_forward_rejects_invalid_initial_train_size() -> None:
    with pytest.raises(ValueError, match="at least min_observations"):
        walk_forward_backtest(
            sample_prices(),
            initial_train_size=5,
        )


def test_walk_forward_rejects_training_size_larger_than_data() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        walk_forward_backtest(
            sample_prices(10),
            initial_train_size=10,
            min_observations=10,
        )


def test_walk_forward_rejects_non_series_forecaster_output() -> None:
    def invalid_forecaster(
        history: pd.Series,
        horizon: int,
    ) -> list[float]:
        return [float(history.iloc[-1])] * horizon

    with pytest.raises(TypeError, match="pandas Series"):
        walk_forward_backtest(
            sample_prices(),
            initial_train_size=20,
            forecaster=invalid_forecaster,
        )


def test_walk_forward_rejects_wrong_prediction_count() -> None:
    def invalid_forecaster(
        history: pd.Series,
        horizon: int,
    ) -> pd.Series:
        return pd.Series([float(history.iloc[-1])])

    with pytest.raises(ValueError, match="exactly the requested"):
        walk_forward_backtest(
            sample_prices(),
            initial_train_size=20,
            horizon=3,
            forecaster=invalid_forecaster,
        )


def test_walk_forward_rejects_non_finite_predictions() -> None:
    def invalid_forecaster(
        history: pd.Series,
        horizon: int,
    ) -> pd.Series:
        return pd.Series([float("nan")] * horizon)

    with pytest.raises(ValueError, match="non-finite"):
        walk_forward_backtest(
            sample_prices(),
            initial_train_size=20,
            horizon=2,
            forecaster=invalid_forecaster,
        )


def test_walk_forward_rejects_non_positive_predictions() -> None:
    def invalid_forecaster(
        history: pd.Series,
        horizon: int,
    ) -> pd.Series:
        return pd.Series([-1.0] * horizon)

    with pytest.raises(ValueError, match="must be positive"):
        walk_forward_backtest(
            sample_prices(),
            initial_train_size=20,
            horizon=2,
            forecaster=invalid_forecaster,
        )
