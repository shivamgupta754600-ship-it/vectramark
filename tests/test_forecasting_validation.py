"""Tests for forecasting input validation."""

import numpy as np
import pandas as pd
import pytest

from vectramark.domain.forecasting.validation import validate_closing_prices


def valid_prices() -> pd.Series:
    return pd.Series(
        range(100, 110),
        index=pd.date_range("2026-01-01", periods=10, freq="D"),
        dtype=float,
    )


def test_valid_closing_prices_are_accepted() -> None:
    prices = valid_prices()

    result = validate_closing_prices(prices)

    assert len(result) == 10
    assert result.iloc[-1] == 109.0
    assert result is not prices


def test_empty_prices_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_closing_prices(pd.Series(dtype=float))


def test_insufficient_prices_are_rejected() -> None:
    with pytest.raises(ValueError, match="observations are required"):
        validate_closing_prices(pd.Series([100.0, 101.0]))


@pytest.mark.parametrize("invalid_price", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_prices_are_rejected(invalid_price: float) -> None:
    prices = valid_prices()
    prices.iloc[-1] = invalid_price

    with pytest.raises(ValueError):
        validate_closing_prices(prices)


def test_non_chronological_prices_are_rejected() -> None:
    prices = valid_prices().iloc[::-1]

    with pytest.raises(ValueError, match="chronological"):
        validate_closing_prices(prices)


def test_duplicate_timestamps_are_rejected() -> None:
    prices = valid_prices()
    prices.index = [
        pd.Timestamp("2026-01-01") if i == 1 else timestamp
        for i, timestamp in enumerate(prices.index)
    ]

    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_closing_prices(prices)


def test_non_series_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        validate_closing_prices([100.0, 101.0])  # type: ignore[arg-type]
