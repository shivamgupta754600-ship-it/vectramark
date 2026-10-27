"""Response schema for market-news safety checks."""

from typing import Literal

from pydantic import BaseModel, Field

from vectramark.api.schemas.market_data import MarketSymbol


class MarketSafetyResponse(BaseModel):
    """Safety flags for market-news interpretation, not trading signals."""

    symbol: MarketSymbol
    status: Literal["review_required", "no_flags_detected"]
    review_required: bool
    flags: list[str]
    checked_articles: int = Field(ge=0)
    disclaimer: str
