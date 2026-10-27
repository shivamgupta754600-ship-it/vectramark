"""Pydantic schemas for market news responses."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from vectramark.api.schemas.market_data import MarketSymbol


class NewsArticleResponse(BaseModel):
    """A normalized news article."""

    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    source: str | None = Field(default=None, max_length=200)
    published_at: datetime | None = None
    summary: str | None = Field(default=None, max_length=2000)
    publication_time_status: Literal["known", "unknown"]


class MarketNewsResponse(BaseModel):
    """Market news results with freshness and interpretation safeguards."""

    symbol: MarketSymbol
    count: int = Field(ge=0)
    freshness_hours: int = Field(ge=1, le=720)
    data: list[NewsArticleResponse]
    disclaimer: str
