"""Application service for support and resistance analysis."""

from vectramark.application.market_data_service import MarketDataService
from vectramark.core.exceptions import DataValidationError
from vectramark.data.market_config import get_market
from vectramark.domain.analytics.support_resistance import (
    SupportResistanceResult,
    detect_support_resistance,
)


class SupportResistanceService:
    """Coordinate validated market-data retrieval and zone detection."""

    def __init__(self, market_data_service: MarketDataService) -> None:
        self._market_data_service = market_data_service

    def analyze(
        self,
        symbol: str,
        period: str = "1y",
        swing_window: int = 3,
        tolerance_pct: float = 0.5,
        max_levels: int = 10,
    ) -> SupportResistanceResult:
        """Fetch daily OHLC data and detect historical price zones."""

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Market symbol cannot be empty.")

        normalized_symbol = symbol.strip().upper()

        try:
            get_market(normalized_symbol)
        except (ValueError, KeyError) as exc:
            raise ValueError("Unsupported market symbol.") from exc

        if swing_window < 1 or swing_window > 20:
            raise ValueError("swing_window must be between 1 and 20.")

        if not 0.0 <= tolerance_pct <= 5.0:
            raise ValueError("tolerance_pct must be between 0 and 5.")

        if max_levels < 1 or max_levels > 50:
            raise ValueError("max_levels must be between 1 and 50.")

        market_data = self._market_data_service.fetch_and_store(
            symbol=normalized_symbol,
            period=period,
            interval="1d",
        )

        try:
            return detect_support_resistance(
                market_data,
                swing_window=swing_window,
                tolerance_pct=tolerance_pct,
                max_levels=max_levels,
            )
        except (ValueError, TypeError) as exc:
            raise DataValidationError(
                "Historical market data could not be used for "
                "support and resistance analysis."
            ) from exc
