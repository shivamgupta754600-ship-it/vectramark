import pandas as pd
import pytest

from vectramark.application.backtesting_service import BacktestingService


class FakeMarketDataService:
    """Return predictable historical data without a network request."""

    def __init__(self, market_data: pd.DataFrame) -> None:
        self.market_data = market_data.copy()
        self.last_request = None

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


def test_backtest_returns_actuals_predictions_and_metrics() -> None:
    service = BacktestingService(FakeMarketDataService(make_market_data()))

    result = service.backtest(
        symbol="XAUUSD",
        initial_train_size=20,
        horizon=5,
    )

    assert len(result.actual) > 0
    assert len(result.actual) == len(result.predicted)
    assert result.initial_train_size == 20
    assert result.horizon == 5
    assert {"mae", "rmse", "mape"} <= set(result.metrics)


def test_backtest_normalizes_symbol_and_forwards_parameters() -> None:
    fake = FakeMarketDataService(make_market_data())
    service = BacktestingService(fake)

    service.backtest(
        symbol=" xauusd ",
        period="6mo",
        interval="1d",
        initial_train_size=20,
        horizon=5,
    )

    assert fake.last_request == {
        "symbol": "XAUUSD",
        "period": "6mo",
        "interval": "1d",
    }


def test_backtest_rejects_training_size_beyond_available_history() -> None:
    service = BacktestingService(FakeMarketDataService(make_market_data(count=20)))

    with pytest.raises(ValueError, match="at least one observation"):
        service.backtest(
            symbol="XAUUSD",
            initial_train_size=20,
            horizon=5,
        )


def test_backtest_rejects_invalid_horizon() -> None:
    service = BacktestingService(FakeMarketDataService(make_market_data()))

    with pytest.raises(ValueError, match="horizon"):
        service.backtest(
            symbol="XAUUSD",
            initial_train_size=20,
            horizon=0,
        )


def test_backtest_rejects_missing_close_column() -> None:
    market_data = pd.DataFrame(
        {"Open": [100.0] * 40},
        index=pd.date_range(
            start="2026-01-01",
            periods=40,
            freq="B",
        ),
    )

    service = BacktestingService(FakeMarketDataService(market_data))

    with pytest.raises(ValueError, match="Close"):
        service.backtest(
            symbol="XAUUSD",
            initial_train_size=20,
            horizon=5,
        )
