from functools import lru_cache

from fastapi import Depends

from vectramark.application.market_data_quality_service import (
    MarketDataQualityService,
)
from vectramark.application.market_data_service import MarketDataService
from vectramark.infrastructure.market_data.yfinance_provider import (
    YahooFinanceProvider,
)
from vectramark.infrastructure.storage.market_store import LocalMarketStore


@lru_cache(maxsize=1)
def get_market_data_service() -> MarketDataService:
    """Return the shared market-data application service."""

    provider = YahooFinanceProvider()
    store = LocalMarketStore()

    return MarketDataService(
        provider=provider,
        store=store,
    )


def get_market_data_quality_service(
    market_data_service: MarketDataService = Depends(get_market_data_service),
) -> MarketDataQualityService:
    """Build the quality service using the shared market-data service."""

    return MarketDataQualityService(
        market_data_service=market_data_service,
    )
