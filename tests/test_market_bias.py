"""Tests for VectraMark market-bias analysis."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vectramark.api.dependencies import get_market_data_service
from vectramark.api.main import app
from vectramark.application.market_bias_service import MarketBiasService
from vectramark.domain.analytics.market_bias import calculate_market_bias


class FakeMarketDataService:
    """Return deterministic historical data without network access."""

    def __init__(self, market_data: pd.DataFrame) -> None:
        self.market_data = market_data.copy()
        self.last_request: dict[str, str] | None = None

    def fetch_and_store(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.last_request = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
        }
        return self.market_data.copy()


def make_market_data(
    count: int = 300,
    flat: bool = False,
) -> pd.DataFrame:
    """Create positive, deterministic daily prices."""

    dates = pd.date_range(
        start="2025-01-01",
        periods=count,
        freq="B",
    )

    if flat:
        prices = np.full(count, 100.0)
    else:
        index = np.arange(count, dtype=float)
        prices = 100.0 + (0.2 * index) + (0.3 * np.sin(index / 4.0))

    return pd.DataFrame(
        {"Close": prices},
        index=dates,
    )


def test_market_bias_returns_explainable_signals() -> None:
    prices = make_market_data()["Close"]

    result = calculate_market_bias(prices, timeframe="daily")

    assert result.bias in {"bullish", "bearish", "neutral"}
    assert -5 <= result.score <= 5
    assert result.latest_close == pytest.approx(prices.iloc[-1])
    assert result.observations == len(prices)
    assert len(result.signals) == 5

    signal_names = {signal.name for signal in result.signals}

    assert signal_names == {
        "price_vs_sma20",
        "price_vs_ema20",
        "rsi14",
        "macd_histogram",
        "horizon_return",
    }

    assert all(signal.explanation for signal in result.signals)


def test_flat_prices_produce_neutral_bias() -> None:
    prices = make_market_data(count=60, flat=True)["Close"]

    result = calculate_market_bias(prices, timeframe="daily")

    assert result.bias == "neutral"
    assert result.score == 0
    assert all(signal.direction == "neutral" for signal in result.signals)


def test_yearly_return_is_unavailable_without_enough_history() -> None:
    prices = make_market_data(count=100)["Close"]

    result = calculate_market_bias(prices, timeframe="yearly")

    yearly_signal = next(
        signal for signal in result.signals if signal.name == "horizon_return"
    )

    assert yearly_signal.direction == "unavailable"
    assert yearly_signal.value is None
    assert "253" in yearly_signal.explanation


def test_market_bias_rejects_missing_closing_prices() -> None:
    prices = pd.Series([100.0, np.nan, 102.0])

    with pytest.raises(ValueError, match="missing values"):
        calculate_market_bias(prices)


def test_market_bias_rejects_non_positive_prices() -> None:
    prices = pd.Series([100.0, 0.0, 102.0])

    with pytest.raises(ValueError, match="greater than zero"):
        calculate_market_bias(prices)


def test_market_bias_rejects_invalid_timeframe() -> None:
    prices = make_market_data()["Close"]

    with pytest.raises(ValueError, match="timeframe"):
        calculate_market_bias(prices, timeframe="quarterly")  # type: ignore[arg-type]


def test_service_normalizes_symbol_and_uses_daily_data() -> None:
    fake = FakeMarketDataService(make_market_data())
    service = MarketBiasService(fake)

    result = service.analyze(
        symbol=" xauusd ",
        period="5y",
        timeframe="yearly",
    )

    assert result.timeframe == "yearly"
    assert fake.last_request == {
        "symbol": "XAUUSD",
        "period": "5y",
        "interval": "1d",
    }


def test_market_bias_api_returns_signals() -> None:
    fake = FakeMarketDataService(make_market_data())

    previous_override = app.dependency_overrides.get(get_market_data_service)
    app.dependency_overrides[get_market_data_service] = lambda: fake

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/analytics/bias/XAUUSD",
                params={
                    "period": "5y",
                    "timeframe": "yearly",
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["symbol"] == "XAUUSD"
        assert body["period"] == "5y"
        assert body["timeframe"] == "yearly"
        assert body["observations"] == 300
        assert body["bias"] in {"bullish", "bearish", "neutral"}
        assert -5 <= body["score"] <= 5
        assert body["latest_close"] > 0
        assert len(body["signals"]) == 5
        assert body["disclaimer"]

        assert fake.last_request == {
            "symbol": "XAUUSD",
            "period": "5y",
            "interval": "1d",
        }

    finally:
        if previous_override is None:
            app.dependency_overrides.pop(
                get_market_data_service,
                None,
            )
        else:
            app.dependency_overrides[get_market_data_service] = previous_override


def test_market_bias_api_rejects_unknown_symbol() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/analytics/bias/UNKNOWN")

    assert response.status_code == 404
