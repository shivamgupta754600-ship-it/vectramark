"""Tests for VectraMark's financial analytics engine."""

import numpy as np
import pandas as pd
import pytest

from vectramark.domain.analytics.analytics_service import AnalyticsService
from vectramark.domain.analytics.returns import (
    calculate_cumulative_returns,
    calculate_daily_returns,
)
from vectramark.domain.analytics.technical_indicators import (
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)
from vectramark.domain.analytics.volatility import (
    calculate_historical_volatility,
)


@pytest.fixture
def close_prices() -> pd.Series:
    """Provide deterministic positive closing prices."""
    return pd.Series(
        [100, 102, 101, 104, 106, 105, 108, 110, 109, 112] * 4,
        dtype=float,
        name="Close",
    )


def test_sma_calculation(close_prices: pd.Series) -> None:
    result = calculate_sma(close_prices, window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx((100 + 102 + 101) / 3)
    assert result.name == "SMA_3"


def test_ema_calculation(close_prices: pd.Series) -> None:
    result = calculate_ema(close_prices, window=3)

    assert result.iloc[:2].isna().all()
    assert np.isfinite(result.iloc[2])
    assert result.name == "EMA_3"


def test_rsi_stays_between_zero_and_one_hundred(
    close_prices: pd.Series,
) -> None:
    result = calculate_rsi(close_prices, window=3)
    valid_values = result.dropna()

    assert not valid_values.empty
    assert (valid_values >= 0).all()
    assert (valid_values <= 100).all()


def test_rsi_is_100_for_strictly_rising_prices() -> None:
    prices = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)

    result = calculate_rsi(prices, window=3)

    assert result.dropna().iloc[-1] == pytest.approx(100)


def test_rsi_is_50_for_flat_prices() -> None:
    prices = pd.Series([10.0] * 8)

    result = calculate_rsi(prices, window=3)

    assert result.dropna().iloc[-1] == pytest.approx(50)


def test_macd_returns_expected_columns(
    close_prices: pd.Series,
) -> None:
    result = calculate_macd(close_prices)

    assert list(result.columns) == [
        "MACD",
        "MACD_signal",
        "MACD_histogram",
    ]

    valid_histogram = result["MACD_histogram"].dropna()
    assert not valid_histogram.empty

    last = result.dropna().iloc[-1]
    assert last["MACD_histogram"] == pytest.approx(last["MACD"] - last["MACD_signal"])


def test_macd_rejects_invalid_window_order() -> None:
    prices = pd.Series([100.0, 101.0, 102.0])

    with pytest.raises(ValueError, match="fast must be smaller than slow"):
        calculate_macd(prices, fast=26, slow=12)


def test_historical_volatility_is_non_negative(
    close_prices: pd.Series,
) -> None:
    result = calculate_historical_volatility(
        close_prices,
        window=3,
    )

    valid_values = result.dropna()

    assert not valid_values.empty
    assert (valid_values >= 0).all()


def test_daily_returns_calculation() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])

    result = calculate_daily_returns(prices)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.10)
    assert result.iloc[2] == pytest.approx(0.10)


def test_cumulative_returns_compound_correctly() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])

    result = calculate_cumulative_returns(prices)

    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.10)
    assert result.iloc[2] == pytest.approx(0.21)


@pytest.mark.parametrize("window", [0, -1])
def test_sma_rejects_invalid_window(window: int) -> None:
    prices = pd.Series([100.0, 101.0, 102.0])

    with pytest.raises(ValueError, match="window must be greater than zero"):
        calculate_sma(prices, window=window)


def test_sma_rejects_boolean_window() -> None:
    prices = pd.Series([100.0, 101.0, 102.0])

    with pytest.raises(TypeError, match="window must be an integer"):
        calculate_sma(prices, window=True)


def test_indicators_reject_missing_prices() -> None:
    prices = pd.Series([100.0, np.nan, 102.0])

    with pytest.raises(ValueError, match="missing values"):
        calculate_sma(prices, window=2)

    with pytest.raises(ValueError, match="missing values"):
        calculate_daily_returns(prices)


def test_indicators_reject_non_positive_prices() -> None:
    prices = pd.Series([100.0, 0.0, 102.0])

    with pytest.raises(ValueError, match="greater than zero"):
        calculate_ema(prices, window=2)

    with pytest.raises(ValueError, match="greater than zero"):
        calculate_historical_volatility(prices, window=2)


def test_analytics_service_combines_calculations(
    close_prices: pd.Series,
) -> None:
    market_data = pd.DataFrame({"Close": close_prices})
    service = AnalyticsService()

    result = service.analyze(
        market_data,
        sma_window=3,
        ema_window=3,
        rsi_window=3,
        volatility_window=3,
    )

    expected_columns = {
        "Close",
        "SMA_3",
        "EMA_3",
        "RSI_3",
        "MACD",
        "MACD_signal",
        "MACD_histogram",
        "Historical_Volatility_3",
        "Daily_Return",
        "Cumulative_Return",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == len(market_data)
    assert result["Cumulative_Return"].iloc[0] == pytest.approx(0.0)


def test_analytics_service_requires_close_column() -> None:
    service = AnalyticsService()

    with pytest.raises(ValueError, match="Close column"):
        service.analyze(pd.DataFrame({"Open": [100.0]}))


def test_analytics_service_rejects_empty_dataframe() -> None:
    service = AnalyticsService()

    with pytest.raises(ValueError, match="cannot be empty"):
        service.analyze(pd.DataFrame({"Close": []}))
