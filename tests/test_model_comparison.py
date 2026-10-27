"""Tests for forecasting baseline comparison."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vectramark.api.backtesting_dependencies import get_backtesting_service
from vectramark.api.main import app
from vectramark.application.backtesting_service import BacktestingService
from vectramark.domain.forecasting.model_comparison import (
    compare_forecasters,
    moving_average_forecast,
)


class FakeMarketDataService:
    """Supply deterministic historical data without network requests."""

    def __init__(self, market_data: pd.DataFrame) -> None:
        self.market_data = market_data.copy()
        self.fetch_count = 0

    def fetch_and_store(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.fetch_count += 1
        return self.market_data.copy()


def make_market_data(count: int = 40) -> pd.DataFrame:
    dates = pd.date_range(
        start="2026-01-01",
        periods=count,
        freq="B",
    )

    return pd.DataFrame(
        {"Close": [100.0 + index for index in range(count)]},
        index=dates,
    )


def test_moving_average_forecast_uses_latest_window() -> None:
    prices = pd.Series(
        [float(value) for value in range(1, 11)],
        index=pd.date_range("2026-01-01", periods=10, freq="D"),
    )

    predictions = moving_average_forecast(
        prices,
        horizon=3,
        window=5,
    )

    assert predictions.tolist() == [8.0, 8.0, 8.0]
    assert predictions.name == "forecast"


def test_model_comparison_uses_identical_evaluation_timestamps() -> None:
    prices = make_market_data()["Close"]

    result = compare_forecasters(
        closing_prices=prices,
        initial_train_size=20,
        horizon=5,
    )

    assert result.naive_result.actual.index.equals(
        result.moving_average_result.actual.index
    )
    assert len(result.naive_result.actual) == len(result.moving_average_result.actual)
    assert result.winner in {
        "naive_baseline",
        "moving_average_baseline",
        "tie",
    }

    for backtest in (
        result.naive_result,
        result.moving_average_result,
    ):
        assert len(backtest.actual) == len(backtest.predicted)
        assert {"mae", "rmse", "mape"} <= set(backtest.metrics)
        assert all(value >= 0 for value in backtest.metrics.values())


def test_model_comparison_endpoint_returns_both_models() -> None:
    fake_market_service = FakeMarketDataService(make_market_data())
    service = BacktestingService(fake_market_service)

    previous_override = app.dependency_overrides.get(get_backtesting_service)
    app.dependency_overrides[get_backtesting_service] = lambda: service

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/backtesting/compare/history/XAUUSD",
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
        assert body["count"] > 0
        assert body["winner"] in {
            "naive_baseline",
            "moving_average_baseline",
            "tie",
        }

        naive = body["naive_baseline"]
        moving_average = body["moving_average_baseline"]

        assert naive["model_name"] == "naive_baseline"
        assert moving_average["model_name"] == "moving_average_baseline"

        assert naive["count"] == moving_average["count"] == body["count"]
        assert len(naive["data"]) == len(moving_average["data"])

        assert {"mae", "rmse", "mape"} <= set(naive["metrics"])
        assert {"mae", "rmse", "mape"} <= set(moving_average["metrics"])

        naive_timestamps = [point["timestamp"] for point in naive["data"]]
        moving_average_timestamps = [
            point["timestamp"] for point in moving_average["data"]
        ]
        assert naive_timestamps == moving_average_timestamps

        # The comparison should fetch the historical data only once.
        assert fake_market_service.fetch_count == 1

    finally:
        if previous_override is None:
            app.dependency_overrides.pop(
                get_backtesting_service,
                None,
            )
        else:
            app.dependency_overrides[get_backtesting_service] = previous_override


@pytest.mark.parametrize("invalid_horizon", [0, -1, True])
def test_moving_average_rejects_invalid_horizon(
    invalid_horizon: int | bool,
) -> None:
    prices = make_market_data()["Close"]

    with pytest.raises((TypeError, ValueError)):
        moving_average_forecast(
            prices,
            horizon=invalid_horizon,
        )
