from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MarketDataQualityChecks(BaseModel):
    """Individual market-data quality check results."""

    model_config = ConfigDict(extra="forbid")

    ohlc_valid: bool
    timestamps_valid: bool
    duplicates_absent: bool
    sufficient_history: bool
    freshness: bool


class MarketDataQualityResponse(BaseModel):
    """API response describing the quality of retrieved market data."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    period: str
    interval: str
    checked_at: datetime
    candle_count: int = Field(ge=0)
    last_candle_timestamp: datetime | None
    data_age_seconds: float | None = Field(default=None, ge=0)
    status: Literal["good", "warning", "invalid"]
    checks: MarketDataQualityChecks
    issues: list[str]
    warnings: list[str]
