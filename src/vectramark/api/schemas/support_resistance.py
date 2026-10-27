"""Pydantic response schemas for support and resistance analysis."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from vectramark.api.schemas.market_data import MarketPeriod


class SupportResistanceLevelResponse(BaseModel):
    """One detected historical support or resistance zone."""

    price: float = Field(gt=0)
    zone_low: float = Field(gt=0)
    zone_high: float = Field(gt=0)
    kind: Literal["support", "resistance", "near_price"]
    touch_count: int = Field(ge=1)
    explanation: str


class SupportResistanceResponse(BaseModel):
    """Support and resistance analysis response."""

    symbol: str
    period: MarketPeriod
    as_of: datetime
    latest_close: float = Field(gt=0)
    observations: int = Field(ge=1)
    swing_window: int = Field(ge=1, le=20)
    tolerance_pct: float = Field(ge=0, le=5)
    levels: list[SupportResistanceLevelResponse]
    disclaimer: str
