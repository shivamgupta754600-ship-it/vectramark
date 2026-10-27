"""Tests for the financial analytics API."""

from collections.abc import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vectramark.api.dependencies import get_market_data_service
from vectramark.api.main import app
from vectramark.core.exceptions import MarketDataError


class FakeMarketDataService:
    """Provide deterministic market data without network requests."""

    def fetch_and_store(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        if symbol not in {
            "XAUUSD",
            "XAGUSD",
            "NASDAQ100",
            "SP500",
            "DJIA",
        }:
            raise ValueError("Unsupported market symbol.")

        if symbol == "XAUUSD" and period == "5d":
            raise MarketDataError("Simulated provider failure.")

        return pd.DataFrame(
            {
                "Open": [2650.0, 2660.0],
                "High": [2680.0, 2690.0],
                "Low": [2640.0, 2650.0],
                "Close": [2670.0, 2680.0],
                "Volume": [1200.0, 1300.0],
            },
            index=pd.DatetimeIndex(
                [
                    "2026-01-05T00:00:00Z",
                    "2026-01-06T00:00:00Z",
                ],
                name="timestamp",
            ),
        )


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client using the fake market-data service."""
    app.dependency_overrides[get_market_data_service] = lambda: FakeMarketDataService()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_market_data_service, None)


def test_historical_analytics_success(client: TestClient) -> None:
    """Check the response structure and calculated analytics."""
    response = client.get(
        "/api/v1/analytics/history/XAUUSD",
        params={
            "period": "1y",
            "interval": "1d",
            "sma_window": 20,
            "ema_window": 20,
            "rsi_window": 14,
            "volatility_window": 20,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["symbol"] == "XAUUSD"
    assert payload["count"] == 2
    assert len(payload["data"]) == 2

    first_row = payload["data"][0]

    assert first_row["Close"] == 2670.0
    assert first_row["SMA_20"] is None
    assert first_row["EMA_20"] is None
    assert first_row["RSI_14"] is None
    assert first_row["timestamp"].startswith("2026-01-05")


def test_analytics_normalizes_lowercase_symbol(
    client: TestClient,
) -> None:
    """Normalize lowercase symbols to uppercase."""
    response = client.get(
        "/api/v1/analytics/history/xauusd",
    )

    assert response.status_code == 200
    assert response.json()["symbol"] == "XAUUSD"


def test_analytics_rejects_invalid_period(
    client: TestClient,
) -> None:
    """Reject unsupported period values."""
    response = client.get(
        "/api/v1/analytics/history/XAUUSD",
        params={"period": "999y"},
    )

    assert response.status_code == 422


def test_analytics_rejects_invalid_interval(
    client: TestClient,
) -> None:
    """Reject unsupported interval values."""
    response = client.get(
        "/api/v1/analytics/history/XAUUSD",
        params={"interval": "7h"},
    )

    assert response.status_code == 422


def test_analytics_rejects_invalid_window(
    client: TestClient,
) -> None:
    """Reject a moving-average window of zero."""
    response = client.get(
        "/api/v1/analytics/history/XAUUSD",
        params={"sma_window": 0},
    )

    assert response.status_code == 422


def test_analytics_provider_failure_returns_502(
    client: TestClient,
) -> None:
    """Return a gateway error when the provider fails."""
    response = client.get(
        "/api/v1/analytics/history/XAUUSD",
        params={"period": "5d"},
    )

    assert response.status_code == 502


def test_analytics_returns_cumulative_return(
    client: TestClient,
) -> None:
    """Check cumulative returns are calculated correctly."""
    response = client.get(
        "/api/v1/analytics/history/XAUUSD",
    )

    assert response.status_code == 200

    data = response.json()["data"]

    assert data[0]["Cumulative_Return"] == pytest.approx(0.0)
    assert data[1]["Cumulative_Return"] == pytest.approx(2680.0 / 2670.0 - 1)
