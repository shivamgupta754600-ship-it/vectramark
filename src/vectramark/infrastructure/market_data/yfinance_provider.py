from collections.abc import Callable

import pandas as pd
import yfinance as yf

from vectramark.core.exceptions import MarketDataError
from vectramark.infrastructure.market_data.provider import (
    MarketDataProvider,
)
from vectramark.infrastructure.market_data.symbol_mapping import (
    get_provider_instrument,
)

ALLOWED_PERIODS = frozenset(
    {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }
)

ALLOWED_INTERVALS = frozenset(
    {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }
)


class YahooFinanceProvider(MarketDataProvider):
    """Download historical market data through Yahoo Finance."""

    def __init__(
        self,
        downloader: Callable[..., pd.DataFrame] | None = None,
        timeout_seconds: int = 15,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 60
        ):
            raise ValueError("timeout_seconds must be an integer between 1 and 60.")

        self._downloader = downloader or yf.download
        self._timeout_seconds = timeout_seconds

    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for an internal market symbol."""
        if not isinstance(period, str) or period not in ALLOWED_PERIODS:
            raise MarketDataError("Unsupported historical data period.")

        if not isinstance(interval, str) or interval not in ALLOWED_INTERVALS:
            raise MarketDataError("Unsupported market data interval.")

        instrument = get_provider_instrument(symbol)

        try:
            data = self._downloader(
                tickers=instrument.provider_symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise MarketDataError(
                f"Historical data provider failed for {instrument.internal_symbol}."
            ) from exc

        if not isinstance(data, pd.DataFrame):
            raise MarketDataError(
                "Historical data provider returned an invalid response."
            )

        if data.empty:
            raise MarketDataError(
                f"No historical data returned for {instrument.internal_symbol}."
            )

        result = data.copy()

        # Yahoo Finance can return either flat or MultiIndex columns.
        if isinstance(result.columns, pd.MultiIndex):
            expected_columns = {
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            }

            normalized_columns: list[str] = []

            for column in result.columns:
                parts = [str(part) for part in column]

                matching = next(
                    (part for part in parts if part in expected_columns),
                    parts[0],
                )
                normalized_columns.append(matching)

            result.columns = normalized_columns
        else:
            result.columns = [str(column) for column in result.columns]

        # A provider may return duplicate column labels. Do not silently
        # select one when the meaning of the data is ambiguous.
        if not result.columns.is_unique:
            raise MarketDataError(
                "Historical data contains ambiguous duplicate columns."
            )

        return result
