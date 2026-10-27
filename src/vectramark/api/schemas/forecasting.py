"""Pydantic response schemas for forecasting endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from vectramark.api.schemas.market_data import (
    MarketInterval,
    MarketPeriod,
    MarketSymbol,
)


class ForecastPointResponse(BaseModel):
    """One predicted closing-price point."""

    step: int = Field(ge=1)
    timestamp: datetime
    predicted_close: float = Field(gt=0)


class ForecastResponse(BaseModel):
    """Standard API response for baseline forecasting."""

    symbol: MarketSymbol
    period: MarketPeriod
    interval: MarketInterval

    horizon: int = Field(ge=1, le=100)

    latest_observed_at: datetime
    latest_observed_close: float = Field(gt=0)

    model_name: Literal["naive_baseline"] = "naive_baseline"
    model_description: str = (
        "Repeats the latest observed closing price for every forecast step."
    )
    timestamp_note: str = (
        "Forecast timestamps are interval-based estimates. "
        "Exchange sessions and holidays are not modelled."
    )

    count: int = Field(ge=1)
    data: list[ForecastPointResponse]
