"""API schemas for explainable market-news context."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from vectramark.api.schemas.market_data import MarketSymbol


class NewsContextArticleResponse(BaseModel):
    """One news article and its context analysis."""

    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2000)
    source: str | None = Field(default=None, max_length=200)
    published_at: datetime | None = None
    publication_time_status: Literal["known", "unknown"]
    theme: str
    confidence: Literal["low", "medium", "high"]
    evidence: list[str]
    market_relevance: str
    interpretation: str


class MarketNewsContextResponse(BaseModel):
    """Market-news context results and their limitations."""

    symbol: MarketSymbol
    count: int = Field(ge=0)
    freshness_hours: int = Field(ge=1, le=720)
    data: list[NewsContextArticleResponse]
    confidence_note: str
    disclaimer: str
