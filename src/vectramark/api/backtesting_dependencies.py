"""Dependencies for the backtesting API."""

from vectramark.api.dependencies import get_market_data_service
from vectramark.application.backtesting_service import BacktestingService


def get_backtesting_service() -> BacktestingService:
    """Create a backtesting service using shared market-data infrastructure."""

    return BacktestingService(
        market_data_service=get_market_data_service(),
    )
