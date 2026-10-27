"""Dependencies for the analytics API."""

from functools import lru_cache

from fastapi import Depends

from vectramark.api.dependencies import (
    get_market_data_quality_service,
    get_market_data_service,
)
from vectramark.application.market_bias_service import MarketBiasService
from vectramark.application.market_data_quality_service import (
    MarketDataQualityService,
)
from vectramark.application.market_data_service import (
    MarketDataService,
)
from vectramark.application.market_research_snapshot_service import (
    MarketResearchSnapshotService,
)
from vectramark.application.market_safety_service import MarketSafetyService
from vectramark.application.news_context_service import NewsContextService
from vectramark.application.support_resistance_service import (
    SupportResistanceService,
)
from vectramark.domain.analytics.analytics_service import AnalyticsService


@lru_cache(maxsize=1)
def get_analytics_service() -> AnalyticsService:
    """Return a reusable analytics service."""
    return AnalyticsService()


def get_market_research_snapshot_service(
    market_data_service: MarketDataService = Depends(get_market_data_service),
    market_data_quality_service: MarketDataQualityService = Depends(
        get_market_data_quality_service
    ),
) -> MarketResearchSnapshotService:
    """Build the service that coordinates market research sections."""
    context_service = NewsContextService()

    return MarketResearchSnapshotService(
        bias_service=MarketBiasService(market_data_service),
        support_resistance_service=SupportResistanceService(market_data_service),
        news_context_service=context_service,
        market_safety_service=MarketSafetyService(context_service=context_service),
        market_data_quality_service=market_data_quality_service,
    )
