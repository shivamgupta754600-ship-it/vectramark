"""Application service for market-bias analysis."""

import pandas as pd

from vectramark.application.market_data_service import MarketDataService
from vectramark.data.market_config import get_market
from vectramark.domain.analytics.market_bias import (
    BiasTimeframe,
    MarketBiasResult,
    calculate_market_bias,
)

SUPPORTED_PERIODS = frozenset({"1mo", "3mo", "6mo", "1y", "2y", "5y"})


class MarketBiasService:
    """Fetch historical prices and coordinate market-bias analysis."""

    def __init__(self, market_data_service: MarketDataService) -> None:
        self._market_data_service = market_data_service

    def analyze(
        self,
        symbol: str,
        period: str = "5y",
        timeframe: BiasTimeframe = "daily",
    ) -> MarketBiasResult:
        """Calculate an explainable bias from historical daily closing prices."""

        if not isinstance(symbol, str):
            raise TypeError("Market symbol must be a string.")

        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("Market symbol cannot be empty.")

        get_market(normalized_symbol)

        if not isinstance(period, str) or period not in SUPPORTED_PERIODS:
            raise ValueError(
                f"period must be one of: {', '.join(sorted(SUPPORTED_PERIODS))}."
            )

        if timeframe not in {"daily", "weekly", "monthly", "yearly"}:
            raise ValueError("timeframe must be daily, weekly, monthly, or yearly.")

        market_data = self._market_data_service.fetch_and_store(
            symbol=normalized_symbol,
            period=period,
            interval="1d",
        )

        if not isinstance(market_data, pd.DataFrame):
            raise ValueError("Market-data service returned invalid data.")

        if market_data.empty:
            raise ValueError("Historical market data cannot be empty.")

        if "Close" not in market_data.columns:
            raise ValueError("Historical market data must contain a Close column.")

        if not isinstance(market_data.index, pd.DatetimeIndex):
            raise ValueError("Market-data index must contain timestamps.")

        market_data = market_data.sort_index()

        if market_data.index.has_duplicates:
            raise ValueError("Historical market data contains duplicate timestamps.")

        closing_prices = pd.to_numeric(
            market_data["Close"].copy(),
            errors="raise",
        )

        return calculate_market_bias(
            closing_prices=closing_prices,
            timeframe=timeframe,
        )
