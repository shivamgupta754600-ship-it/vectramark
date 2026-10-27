"""Tests for support and resistance detection and its API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vectramark.api.dependencies import get_market_data_service
from vectramark.api.main import app
from vectramark.application.support_resistance_service import (
    SupportResistanceService,
)
from vectramark.domain.analytics.support_resistance import (
    detect_support_resistance,
)


class FakeMarketDataService:
    """Provide deterministic OHLC data without external network access."""

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


def make_ohlc_data(count: int = 160) -> pd.DataFrame:
    """Create reproducible prices with recurring highs and lows."""

    index = np.arange(count, dtype=float)
    close = 100.0 + 5.0 * np.sin(index / 5.0)
    high = close + 1.0
    low = close - 1.0

    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(count, 1000.0),
        },
        index=pd.date_range("2025-01-01", periods=count, freq="B"),
    )


def test_detector_returns_valid_levels() -> None:
    result = detect_support_resistance(
        make_ohlc_data(),
        swing_window=3,
        tolerance_pct=1.0,
        max_levels=10,
    )

    assert result.observations == 160
    assert result.latest_close > 0
    assert len(result.levels) <= 10

    for level in result.levels:
        assert level.price > 0
        assert level.zone_low <= level.zone_high
        assert level.touch_count >= 1
        assert level.kind in {"support", "resistance", "near_price"}
        assert level.explanation


def test_detector_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        detect_support_resistance(pd.DataFrame())


def test_detector_rejects_missing_ohlc_columns() -> None:
    data = pd.DataFrame({"Close": [100.0, 101.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        detect_support_resistance(data)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("swing_window", 0, "swing_window"),
        ("swing_window", 21, "swing_window"),
        ("tolerance_pct", 5.1, "tolerance_pct"),
        ("max_levels", 0, "max_levels"),
        ("max_levels", 51, "max_levels"),
    ],
)
def test_detector_rejects_invalid_parameters(
    parameter: str,
    value: int | float,
    message: str,
) -> None:
    kwargs = {
        "swing_window": 3,
        "tolerance_pct": 0.5,
        "max_levels": 10,
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=message):
        detect_support_resistance(make_ohlc_data(), **kwargs)


def test_detector_rejects_non_finite_prices() -> None:
    data = make_ohlc_data()
    data.loc[data.index[5], "High"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        detect_support_resistance(data)


def test_service_normalizes_symbol_and_requests_daily_data() -> None:
    fake = FakeMarketDataService(make_ohlc_data())
    service = SupportResistanceService(fake)

    result = service.analyze(
        symbol=" xauusd ",
        period="1y",
        swing_window=3,
        tolerance_pct=0.5,
        max_levels=10,
    )

    assert result.observations == 160
    assert fake.last_request == {
        "symbol": "XAUUSD",
        "period": "1y",
        "interval": "1d",
    }


def test_support_resistance_api_returns_levels() -> None:
    fake = FakeMarketDataService(make_ohlc_data())
    previous_override = app.dependency_overrides.get(get_market_data_service)
    app.dependency_overrides[get_market_data_service] = lambda: fake

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/analytics/levels/XAUUSD",
                params={
                    "period": "1y",
                    "swing_window": 3,
                    "tolerance_pct": 1.0,
                    "max_levels": 8,
                },
            )

        assert response.status_code == 200

        body = response.json()
        assert body["symbol"] == "XAUUSD"
        assert body["period"] == "1y"
        assert body["observations"] == 160
        assert body["latest_close"] > 0
        assert len(body["levels"]) <= 8
        assert body["disclaimer"]
        assert fake.last_request == {
            "symbol": "XAUUSD",
            "period": "1y",
            "interval": "1d",
        }

    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_market_data_service, None)
        else:
            app.dependency_overrides[get_market_data_service] = previous_override


def test_support_resistance_api_rejects_unknown_symbol() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/analytics/levels/UNKNOWN")

    assert response.status_code == 404


def test_support_resistance_api_rejects_invalid_query_parameters() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/analytics/levels/XAUUSD",
            params={"swing_window": 0},
        )

    assert response.status_code == 422
