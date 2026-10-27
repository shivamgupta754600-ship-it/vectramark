"""Pydantic response schemas for backtesting."""

from datetime import datetime

from pydantic import BaseModel, Field

from vectramark.api.schemas.market_data import (
    MarketInterval,
    MarketPeriod,
    MarketSymbol,
)


class BacktestMetricsResponse(BaseModel):
    """Forecast evaluation metrics."""

    mae: float = Field(ge=0)
    rmse: float = Field(ge=0)
    mape: float = Field(ge=0)


class BacktestPointResponse(BaseModel):
    """Actual and predicted closing prices for one test observation."""

    timestamp: datetime
    actual_close: float = Field(gt=0)
    predicted_close: float = Field(gt=0)
    absolute_error: float = Field(ge=0)


class BacktestResponse(BaseModel):
    """Standard response for historical walk-forward backtesting."""

    symbol: MarketSymbol
    period: MarketPeriod
    interval: MarketInterval

    model_name: str = "naive_baseline"
    model_description: str = (
        "Repeats the most recently observed training price at each forecast step."
    )

    initial_train_size: int = Field(ge=10)
    horizon: int = Field(ge=1, le=100)
    count: int = Field(ge=1)

    metrics: BacktestMetricsResponse
    data: list[BacktestPointResponse]


class BacktestModelResultResponse(BaseModel):
    """Results for one forecasting model."""

    model_name: str
    model_description: str
    count: int = Field(ge=1)
    metrics: BacktestMetricsResponse
    data: list[BacktestPointResponse]


class ModelComparisonResponse(BaseModel):
    """Comparable results for multiple forecasting baselines."""

    symbol: MarketSymbol
    period: MarketPeriod
    interval: MarketInterval
    initial_train_size: int = Field(ge=10)
    horizon: int = Field(ge=1, le=100)
    count: int = Field(ge=1)
    winner: str
    naive_baseline: BacktestModelResultResponse
    moving_average_baseline: BacktestModelResultResponse
