"""Application service for historical financial analytics."""

from __future__ import annotations

import pandas as pd

from vectramark.domain.analytics.returns import (
    calculate_cumulative_returns,
    calculate_daily_returns,
)
from vectramark.domain.analytics.technical_indicators import (
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)
from vectramark.domain.analytics.volatility import (
    calculate_historical_volatility,
)


class AnalyticsService:
    """Coordinate historical market analytics calculations."""

    def analyze(
        self,
        market_data: pd.DataFrame,
        sma_window: int = 20,
        ema_window: int = 20,
        rsi_window: int = 14,
        volatility_window: int = 20,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """Return market data with historical analytics columns."""
        if not isinstance(market_data, pd.DataFrame):
            raise TypeError("market_data must be a pandas DataFrame.")

        if market_data.empty:
            raise ValueError("market_data cannot be empty.")

        if "Close" not in market_data.columns:
            raise ValueError("market_data must contain a Close column.")

        close = market_data["Close"]

        result = market_data.copy()

        result[f"SMA_{sma_window}"] = calculate_sma(
            close,
            window=sma_window,
        )

        result[f"EMA_{ema_window}"] = calculate_ema(
            close,
            window=ema_window,
        )

        result[f"RSI_{rsi_window}"] = calculate_rsi(
            close,
            window=rsi_window,
        )

        macd = calculate_macd(close)
        result = result.join(macd)

        result[f"Historical_Volatility_{volatility_window}"] = (
            calculate_historical_volatility(
                close,
                window=volatility_window,
                periods_per_year=periods_per_year,
            )
        )

        result["Daily_Return"] = calculate_daily_returns(close)
        result["Cumulative_Return"] = calculate_cumulative_returns(close)

        return result
