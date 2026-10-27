"""API routes for baseline market-price forecasting."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from vectramark.api.forecasting_dependencies import get_forecasting_service
from vectramark.api.schemas.forecasting import (
    ForecastPointResponse,
    ForecastResponse,
)
from vectramark.api.schemas.market_data import MarketInterval, MarketPeriod
from vectramark.application.forecasting_service import ForecastingService
from vectramark.core.exceptions import DataValidationError, MarketDataError
from vectramark.data.market_config import get_market

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/forecasting",
    tags=["Forecasting"],
)


@router.get(
    "/history/{symbol}",
    response_model=ForecastResponse,
    summary="Forecast future closing prices",
    description=(
        "Generate a baseline forecast from historical closing prices. "
        "The naive baseline repeats the most recent closing price. "
        "Forecast timestamps are estimates and do not model market sessions "
        "or exchange holidays."
    ),
)
def get_forecast_history(
    symbol: str,
    period: MarketPeriod = Query(default="1y"),
    interval: MarketInterval = Query(default="1d"),
    horizon: int = Query(default=5, ge=1, le=100),
    service: ForecastingService = Depends(get_forecasting_service),
) -> ForecastResponse:
    """Generate a baseline forecast for a supported market."""

    normalized_symbol = symbol.strip().upper()

    if not normalized_symbol:
        raise HTTPException(
            status_code=400,
            detail="Market symbol cannot be empty.",
        )

    try:
        get_market(normalized_symbol)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Unsupported market symbol.",
        ) from exc

    try:
        result = service.forecast(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
            horizon=horizon,
        )

        forecast_points = [
            ForecastPointResponse(
                step=point.step,
                timestamp=point.timestamp.to_pydatetime(),
                predicted_close=point.predicted_close,
            )
            for point in result.points
        ]

        return ForecastResponse(
            symbol=result.symbol,
            period=result.period,
            interval=result.interval,
            horizon=result.horizon,
            latest_observed_at=result.latest_observed_at.to_pydatetime(),
            latest_observed_close=result.latest_observed_close,
            count=len(forecast_points),
            data=forecast_points,
        )

    except (MarketDataError, DataValidationError) as exc:
        logger.warning(
            "Forecasting data failure for symbol %s: %s",
            normalized_symbol,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=502,
            detail="Market data could not be retrieved or validated.",
        ) from exc

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected forecasting failure for symbol %s",
            normalized_symbol,
        )

        raise HTTPException(
            status_code=502,
            detail="Unable to generate the market forecast.",
        ) from exc
