"""Response schemas for the combined market research snapshot."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from vectramark.api.schemas.market_data import MarketPeriod, MarketSymbol


class ResearchSnapshotSection(BaseModel):
    """Result of one independently executed analysis section."""

    status: Literal["ok", "error"]
    generated_at: datetime
    data: Any = None
    error_code: str | None = None
    message: str | None = None


class MarketResearchSnapshotResponse(BaseModel):
    """Combined research context with independently reported failures."""

    symbol: MarketSymbol
    period: MarketPeriod
    interval: Literal["1d", "1wk", "1mo"]
    timeframe: Literal["daily", "weekly", "monthly", "yearly"]
    requested_at: datetime

    bias: ResearchSnapshotSection
    support_resistance: ResearchSnapshotSection
    news_context: ResearchSnapshotSection
    news_safety: ResearchSnapshotSection
    data_quality: ResearchSnapshotSection

    disclaimer: str = Field(min_length=1)
