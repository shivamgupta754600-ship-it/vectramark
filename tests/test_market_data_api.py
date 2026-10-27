from collections.abc import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vectramark.api.dependencies import get_market_data_service
from vectramark.api.main import app
from vectramark.core.exceptions import MarketDataError


class FakeMarketDataService:
    """Fake service that avoids external network requests."""

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
    """Create a test client with an isolated fake service."""

    app.dependency_overrides[get_market_data_service] = lambda: FakeMarketDataService()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_market_data_service, None)


def test_historical_market_data_success(client: TestClient) -> None:
    response = client.get(
        "/api/v1/market-data/history/XAUUSD",
        params={"period": "1y", "interval": "1d"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["symbol"] == "XAUUSD"
    assert payload["period"] == "1y"
    assert payload["interval"] == "1d"
    assert payload["count"] == 2
    assert len(payload["data"]) == 2
    assert payload["data"][0]["open"] == 2650.0
    assert payload["data"][0]["close"] == 2670.0


def test_historical_market_data_normalizes_symbol(client: TestClient) -> None:
    response = client.get("/api/v1/market-data/history/xauusd")

    assert response.status_code == 200
    assert response.json()["symbol"] == "XAUUSD"


def test_unknown_symbol_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/market-data/history/UNKNOWN")

    assert response.status_code == 404


def test_invalid_period_returns_422(client: TestClient) -> None:
    response = client.get(
        "/api/v1/market-data/history/XAUUSD",
        params={"period": "999y"},
    )

    assert response.status_code == 422


def test_invalid_interval_returns_422(client: TestClient) -> None:
    response = client.get(
        "/api/v1/market-data/history/XAUUSD",
        params={"interval": "7h"},
    )

    assert response.status_code == 422


def test_provider_failure_returns_502(client: TestClient) -> None:
    response = client.get(
        "/api/v1/market-data/history/XAUUSD",
        params={"period": "5d"},
    )

    assert response.status_code == 502


def test_supported_symbols_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/market-data/supported-symbols")

    assert response.status_code == 200
    assert set(response.json()) == {
        "XAUUSD",
        "XAGUSD",
        "NASDAQ100",
        "SP500",
        "DJIA",
    }


def test_quality_endpoint_returns_typed_quality_report(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/market-data/quality/XAUUSD",
        params={"period": "1y", "interval": "1d"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["symbol"] == "XAUUSD"
    assert payload["period"] == "1y"
    assert payload["interval"] == "1d"
    assert payload["candle_count"] == 2
    assert payload["status"] in {"good", "warning", "invalid"}
    assert payload["checks"]["ohlc_valid"] is True
    assert payload["checks"]["timestamps_valid"] is True
    assert payload["checks"]["duplicates_absent"] is True
    assert payload["checks"]["sufficient_history"] is False
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["issues"], list)
    assert payload["last_candle_timestamp"] is not None


def test_quality_endpoint_normalizes_symbol(client: TestClient) -> None:
    response = client.get("/api/v1/market-data/quality/xauusd")

    assert response.status_code == 200
    assert response.json()["symbol"] == "XAUUSD"


def test_quality_endpoint_unknown_symbol_returns_404(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/market-data/quality/UNKNOWN")

    assert response.status_code == 404


def test_quality_endpoint_invalid_period_returns_422(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/market-data/quality/XAUUSD",
        params={"period": "999y"},
    )

    assert response.status_code == 422


def test_quality_endpoint_invalid_interval_returns_422(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/market-data/quality/XAUUSD",
        params={"interval": "7h"},
    )

    assert response.status_code == 422


def test_quality_endpoint_provider_failure_returns_502(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/market-data/quality/XAUUSD",
        params={"period": "5d"},
    )

    assert response.status_code == 502
