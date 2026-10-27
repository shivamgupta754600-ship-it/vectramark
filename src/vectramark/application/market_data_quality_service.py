from typing import Any

from vectramark.application.market_data_service import MarketDataService
from vectramark.domain.market_data.quality import MarketDataQualityAnalyzer


class MarketDataQualityService:
    """Coordinate market-data retrieval and quality assessment."""

    def __init__(
        self,
        market_data_service: MarketDataService,
        analyzer: MarketDataQualityAnalyzer | None = None,
    ) -> None:
        self._market_data_service = market_data_service
        self._analyzer = analyzer or MarketDataQualityAnalyzer()

    def assess(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> dict[str, Any]:
        """Retrieve validated data, then assess its quality."""

        market_data = self._market_data_service.fetch_and_store(
            symbol=symbol,
            period=period,
            interval=interval,
        )

        return self._analyzer.assess(
            data=market_data,
            interval=interval,
        )
