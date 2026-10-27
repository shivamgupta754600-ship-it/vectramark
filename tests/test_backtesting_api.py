import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vectramark.api.backtesting_dependencies import get_backtesting_service
from vectramark.api.main import app
from vectramark.application.backtesting_service import BacktestingService


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


def make_market_data(count: int = 40) -> pd.DataFrame:
    dates = pd.date_range(
        start="2026-01-01",
        periods=count,
        freq="B",
    )

    return pd.DataFrame(
        {"Close": [100.0 + i for i in range(count)]},
        index=dates,
    )


@pytest.fixture
def client():
    service = BacktestingService(FakeMarketDataService(make_market_data()))

    previous_override = app.dependency_overrides.get(get_backtesting_service)
    app.dependency_overrides[get_backtesting_service] = lambda: service

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(
                get_backtesting_service,
                None,
            )
        else:
            app.dependency_overrides[get_backtesting_service] = previous_override


def test_backtesting_endpoint_returns_evaluation(client) -> None:
    response = client.get(
        "/api/v1/backtesting/history/XAUUSD",
        params={
            "period": "1y",
            "interval": "1d",
            "initial_train_size": 20,
            "horizon": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["symbol"] == "XAUUSD"
    assert body["model_name"] == "naive_baseline"
    assert body["initial_train_size"] == 20
    assert body["horizon"] == 5
    assert body["count"] > 0
    assert len(body["data"]) == body["count"]

    assert {"mae", "rmse", "mape"} <= set(body["metrics"])

    point = body["data"][0]
    assert point["actual_close"] > 0
    assert point["predicted_close"] > 0
    assert point["absolute_error"] >= 0


def test_backtesting_endpoint_rejects_unknown_symbol(client) -> None:
    response = client.get("/api/v1/backtesting/history/UNKNOWN")

    assert response.status_code == 404


def test_backtesting_endpoint_validates_training_size(client) -> None:
    response = client.get(
        "/api/v1/backtesting/history/XAUUSD",
        params={"initial_train_size": 5},
    )

    assert response.status_code == 422


def test_backtesting_endpoint_rejects_insufficient_history(client) -> None:
    response = client.get(
        "/api/v1/backtesting/history/XAUUSD",
        params={
            "initial_train_size": 100,
            "horizon": 5,
        },
    )

    assert response.status_code == 422
