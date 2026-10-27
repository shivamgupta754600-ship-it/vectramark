import pandas as pd

from vectramark.application.market_data_quality_service import (
    MarketDataQualityService,
)


class FakeMarketDataService:
    """Fake market-data service for isolated unit tests."""

    def __init__(self) -> None:
        self.received_args: dict[str, str] = {}
        self.data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [103.0, 104.0],
                "Low": [99.0, 100.0],
                "Close": [102.0, 103.0],
            },
            index=pd.date_range(
                "2026-01-01",
                periods=2,
                freq="D",
                tz="UTC",
            ),
        )

    def fetch_and_store(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.received_args = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
        }
        return self.data


class FakeQualityAnalyzer:
    """Fake analyzer to verify service orchestration."""

    def __init__(self) -> None:
        self.received_data: pd.DataFrame | None = None
        self.received_interval: str | None = None

    def assess(
        self,
        data: pd.DataFrame,
        interval: str,
    ) -> dict[str, str]:
        self.received_data = data
        self.received_interval = interval
        return {"status": "good"}


def test_quality_service_forwards_parameters_and_data() -> None:
    market_service = FakeMarketDataService()
    analyzer = FakeQualityAnalyzer()

    service = MarketDataQualityService(
        market_data_service=market_service,
        analyzer=analyzer,
    )

    report = service.assess(
        symbol="XAUUSD",
        period="3mo",
        interval="1h",
    )

    assert market_service.received_args == {
        "symbol": "XAUUSD",
        "period": "3mo",
        "interval": "1h",
    }

    assert analyzer.received_data is not None
    pd.testing.assert_frame_equal(
        analyzer.received_data,
        market_service.data,
    )
    assert analyzer.received_interval == "1h"
    assert report == {"status": "good"}


def test_quality_service_uses_default_period_and_interval() -> None:
    market_service = FakeMarketDataService()
    analyzer = FakeQualityAnalyzer()

    service = MarketDataQualityService(
        market_data_service=market_service,
        analyzer=analyzer,
    )

    service.assess(symbol="SP500")

    assert market_service.received_args == {
        "symbol": "SP500",
        "period": "1y",
        "interval": "1d",
    }
    assert analyzer.received_interval == "1d"
