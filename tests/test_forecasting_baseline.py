"""Tests for the naive forecasting baseline."""

import pandas as pd
import pytest

from vectramark.domain.forecasting.baseline import naive_forecast


def valid_prices() -> pd.Series:
    return pd.Series(
        range(100, 110),
        index=pd.date_range("2026-01-01", periods=10, freq="D"),
        dtype=float,
    )


def test_naive_forecast_repeats_last_price() -> None:
    forecast = naive_forecast(valid_prices(), horizon=5)

    assert len(forecast) == 5
    assert forecast.tolist() == [109.0] * 5


def test_naive_forecast_supports_one_step() -> None:
    forecast = naive_forecast(valid_prices())

    assert len(forecast) == 1
    assert forecast.iloc[0] == 109.0


@pytest.mark.parametrize("horizon", [0, -1])
def test_invalid_horizon_is_rejected(horizon: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        naive_forecast(valid_prices(), horizon=horizon)


def test_non_integer_horizon_is_rejected() -> None:
    with pytest.raises(TypeError, match="integer"):
        naive_forecast(valid_prices(), horizon=1.5)  # type: ignore[arg-type]
