import pandas as pd
import pytest

from vectramark.application.forecasting_service import ForecastingService


class FakeMarketDataService:
    """Fake market-data service for isolated forecasting tests."""

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


def make_market_data(count: int = 12) -> pd.DataFrame:
    """Create deterministic historical closing prices."""

    dates = pd.date_range(
        start="2026-01-01",
        periods=count,
        freq="B",
    )

    return pd.DataFrame(
        {"Close": [100.0 + step for step in range(count)]},
        index=dates,
    )


def test_forecast_repeats_last_observed_close() -> None:
    fake_market_service = FakeMarketDataService(make_market_data())
    service = ForecastingService(fake_market_service)

    result = service.forecast(
        symbol="XAUUSD",
        horizon=5,
    )

    assert result.symbol == "XAUUSD"
    assert result.horizon == 5
    assert result.latest_observed_close == 111.0
    assert len(result.points) == 5
    assert all(point.predicted_close == 111.0 for point in result.points)


def test_forecast_normalizes_symbol_and_passes_parameters() -> None:
    fake_market_service = FakeMarketDataService(make_market_data())
    service = ForecastingService(fake_market_service)

    result = service.forecast(
        symbol=" xauusd ",
        period="6mo",
        interval="1d",
        horizon=3,
    )

    assert result.symbol == "XAUUSD"
    assert fake_market_service.last_request == {
        "symbol": "XAUUSD",
        "period": "6mo",
        "interval": "1d",
    }


def test_daily_forecast_timestamps_skip_weekends() -> None:
    fake_market_service = FakeMarketDataService(make_market_data())
    service = ForecastingService(fake_market_service)

    result = service.forecast(
        symbol="XAUUSD",
        horizon=3,
        interval="1d",
    )

    expected = result.latest_observed_at + pd.offsets.BDay(1)

    assert result.points[0].timestamp == expected
    assert all(point.timestamp.weekday() < 5 for point in result.points)


def test_forecast_rejects_zero_horizon() -> None:
    service = ForecastingService(FakeMarketDataService(make_market_data()))

    with pytest.raises(ValueError, match="horizon"):
        service.forecast(
            symbol="XAUUSD",
            horizon=0,
        )


def test_forecast_rejects_unsupported_interval() -> None:
    service = ForecastingService(FakeMarketDataService(make_market_data()))

    with pytest.raises(ValueError, match="interval"):
        service.forecast(
            symbol="XAUUSD",
            interval="invalid",
        )


def test_forecast_rejects_insufficient_history() -> None:
    service = ForecastingService(FakeMarketDataService(make_market_data(count=5)))

    with pytest.raises(ValueError, match="observations"):
        service.forecast(
            symbol="XAUUSD",
            horizon=2,
        )


def test_forecast_rejects_missing_close_column() -> None:
    market_data = pd.DataFrame(
        {
            "Open": [100.0] * 12,
        },
        index=pd.date_range(
            start="2026-01-01",
            periods=12,
            freq="B",
        ),
    )

    service = ForecastingService(FakeMarketDataService(market_data))

    with pytest.raises(ValueError, match="Close"):
        service.forecast(
            symbol="XAUUSD",
            horizon=2,
        )
