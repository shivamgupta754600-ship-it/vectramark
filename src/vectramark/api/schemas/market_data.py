from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

MarketSymbol = Literal[
    "XAUUSD",
    "XAGUSD",
    "NASDAQ100",
    "SP500",
    "DJIA",
]

MarketPeriod = Literal[
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
]

MarketInterval = Literal[
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
]


class MarketCandleResponse(BaseModel):
    """One historical market-data candle."""

    timestamp: datetime

    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)

    volume: float | None = Field(default=None, ge=0)


class HistoricalMarketDataResponse(BaseModel):
    """Standard API response for historical market data."""

    symbol: MarketSymbol
    period: MarketPeriod
    interval: MarketInterval

    count: int = Field(ge=0)

    data: list[MarketCandleResponse]
