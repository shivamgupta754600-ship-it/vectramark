"""Dependencies for forecasting API routes."""

from vectramark.api.dependencies import get_market_data_service
from vectramark.application.forecasting_service import ForecastingService


def get_forecasting_service() -> ForecastingService:
    """Create a forecasting service using the shared market-data service."""

    return ForecastingService(
        market_data_service=get_market_data_service(),
    )
