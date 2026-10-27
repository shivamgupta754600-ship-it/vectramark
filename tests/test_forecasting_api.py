import pandas as pd
from fastapi.testclient import TestClient

from vectramark.api.forecasting_dependencies import get_forecasting_service
from vectramark.api.main import app
from vectramark.application.forecasting_service import ForecastingService


class FakeMarketDataService:
    def __init__(self, market_data: pd.DataFrame) -> None:
        self.market_data = market_data.copy()

    def fetch_and_store(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        return self.market_data.copy()


def make_market_data(count: int = 12) -> pd.DataFrame:
    dates = pd.date_range(
        start="2026-01-01",
        periods=count,
        freq="B",
    )

    return pd.DataFrame(
        {"Close": [100.0 + step for step in range(count)]},
        index=dates,
    )


def create_client(monkeypatch, market_data: pd.DataFrame) -> TestClient:
    service = ForecastingService(FakeMarketDataService(market_data))

    monkeypatch.setitem(
        app.dependency_overrides,
        get_forecasting_service,
        lambda: service,
    )

    return TestClient(app)


def test_forecasting_endpoint_returns_forecast(monkeypatch) -> None:
    client = create_client(
        monkeypatch,
        make_market_data(),
    )

    response = client.get("/api/v1/forecasting/history/XAUUSD?horizon=5")

    assert response.status_code == 200

    body = response.json()

    assert body["symbol"] == "XAUUSD"
    assert body["horizon"] == 5
    assert body["count"] == 5
    assert body["model_name"] == "naive_baseline"
    assert len(body["data"]) == 5
    assert all(point["predicted_close"] == 111.0 for point in body["data"])


def test_forecasting_endpoint_rejects_unknown_symbol(
    monkeypatch,
) -> None:
    client = create_client(
        monkeypatch,
        make_market_data(),
    )

    response = client.get("/api/v1/forecasting/history/UNKNOWN")

    assert response.status_code == 404


def test_forecasting_endpoint_validates_horizon(
    monkeypatch,
) -> None:
    client = create_client(
        monkeypatch,
        make_market_data(),
    )

    response = client.get("/api/v1/forecasting/history/XAUUSD?horizon=0")

    assert response.status_code == 422


def test_forecasting_endpoint_rejects_insufficient_history(
    monkeypatch,
) -> None:
    client = create_client(
        monkeypatch,
        make_market_data(count=5),
    )

    response = client.get("/api/v1/forecasting/history/XAUUSD?horizon=2")

    assert response.status_code == 422
