"""Response schemas for financial analytics."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from vectramark.api.schemas.market_data import MarketPeriod, MarketSymbol


class AnalyticsResponse(BaseModel):
    """Historical market analytics response."""

    symbol: MarketSymbol
    count: int = Field(ge=0)
    sma_window: int = Field(gt=0)
    ema_window: int = Field(gt=0)
    rsi_window: int = Field(gt=0)
    volatility_window: int = Field(gt=0)
    periods_per_year: int = Field(gt=0)
    data: list[dict[str, float | str | None]]


class AnalyticsQuery(BaseModel):
    """Validated analytics query parameters."""

    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y"] = "1y"
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    sma_window: int = Field(default=20, ge=1, le=250)
    ema_window: int = Field(default=20, ge=1, le=250)
    rsi_window: int = Field(default=14, ge=1, le=100)
    volatility_window: int = Field(default=20, ge=2, le=250)
    periods_per_year: int = Field(default=252, ge=1, le=366)


class BiasSignalResponse(BaseModel):
    """One indicator signal included in market-bias analysis."""

    name: str
    value: float | None
    unit: str
    direction: Literal[
        "bullish",
        "bearish",
        "neutral",
        "unavailable",
    ]
    explanation: str


class MarketBiasResponse(BaseModel):
    """API response for rule-based market-bias analysis."""

    symbol: MarketSymbol
    period: MarketPeriod
    timeframe: Literal["daily", "weekly", "monthly", "yearly"]
    as_of: datetime
    latest_close: float = Field(gt=0)
    observations: int = Field(ge=1)
    bias: Literal["bullish", "bearish", "neutral"]
    score: int = Field(ge=-5, le=5)
    explanation: str
    signals: list[BiasSignalResponse]
    disclaimer: str
