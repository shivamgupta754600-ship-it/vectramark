"""API routes for historical walk-forward backtesting."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from vectramark.api.backtesting_dependencies import get_backtesting_service
from vectramark.api.schemas.backtesting import (
    BacktestMetricsResponse,
    BacktestModelResultResponse,
    BacktestPointResponse,
    BacktestResponse,
    ModelComparisonResponse,
)
from vectramark.api.schemas.market_data import MarketInterval, MarketPeriod
from vectramark.application.backtesting_service import BacktestingService
from vectramark.core.exceptions import DataValidationError, MarketDataError
from vectramark.data.market_config import get_market
from vectramark.domain.forecasting.backtesting import BacktestResult

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/backtesting",
    tags=["Backtesting"],
)


@router.get(
    "/history/{symbol}",
    response_model=BacktestResponse,
    summary="Evaluate historical forecast performance",
    description=(
        "Run chronological walk-forward backtesting using the naive baseline "
        "and return actual prices, predictions, and forecast errors."
    ),
)
def get_historical_backtest(
    symbol: str,
    period: MarketPeriod = Query(default="1y"),
    interval: MarketInterval = Query(default="1d"),
    initial_train_size: int = Query(default=60, ge=10, le=5000),
    horizon: int = Query(default=5, ge=1, le=100),
    service: BacktestingService = Depends(get_backtesting_service),
) -> BacktestResponse:
    """Run walk-forward backtesting for a supported market."""

    normalized_symbol = symbol.strip().upper()
    try:
        get_market(normalized_symbol)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Unsupported market symbol.",
        ) from exc

    if not normalized_symbol:
        raise HTTPException(
            status_code=400,
            detail="Market symbol cannot be empty.",
        )

    try:
        result = service.backtest(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
            initial_train_size=initial_train_size,
            horizon=horizon,
        )

        if len(result.actual) == 0:
            raise ValueError("Backtesting produced no evaluation observations.")

        if len(result.actual) != len(result.predicted):
            raise ValueError("Actual and predicted observation counts do not match.")

        points = []

        for timestamp, actual, predicted in zip(
            result.actual.index,
            result.actual.to_numpy(),
            result.predicted.to_numpy(),
            strict=True,
        ):
            actual_close = float(actual)
            predicted_close = float(predicted)

            points.append(
                BacktestPointResponse(
                    timestamp=timestamp.to_pydatetime(),
                    actual_close=actual_close,
                    predicted_close=predicted_close,
                    absolute_error=abs(actual_close - predicted_close),
                )
            )

        metrics = result.metrics

        return BacktestResponse(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
            initial_train_size=result.initial_train_size,
            horizon=result.horizon,
            count=len(points),
            metrics=BacktestMetricsResponse(
                mae=float(metrics["mae"]),
                rmse=float(metrics["rmse"]),
                mape=float(metrics["mape"]),
            ),
            data=points,
        )

    except (MarketDataError, DataValidationError) as exc:
        logger.warning(
            "Backtesting data failure for %s: %s",
            normalized_symbol,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=502,
            detail="Market data could not be retrieved or validated.",
        ) from exc

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected backtesting failure for %s",
            normalized_symbol,
        )

        raise HTTPException(
            status_code=502,
            detail="Unable to evaluate historical forecasts.",
        ) from exc


@router.get(
    "/compare/history/{symbol}",
    response_model=ModelComparisonResponse,
    summary="Compare forecasting baselines",
    description=(
        "Compare naive and moving-average forecasts using chronological "
        "walk-forward evaluation on the same historical observations."
    ),
)
def compare_historical_models(
    symbol: str,
    period: MarketPeriod = Query(default="1y"),
    interval: MarketInterval = Query(default="1d"),
    initial_train_size: int = Query(default=60, ge=10, le=5000),
    horizon: int = Query(default=5, ge=1, le=100),
    service: BacktestingService = Depends(get_backtesting_service),
) -> ModelComparisonResponse:
    """Compare model metrics and historical predictions."""

    normalized_symbol = symbol.strip().upper()

    try:
        get_market(normalized_symbol)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Unsupported market symbol.",
        ) from exc

    if not normalized_symbol:
        raise HTTPException(
            status_code=400,
            detail="Market symbol cannot be empty.",
        )

    try:
        result = service.compare_models(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
            initial_train_size=initial_train_size,
            horizon=horizon,
        )

        def build_model_response(
            model_name: str,
            description: str,
            backtest_result: BacktestResult,
        ) -> BacktestModelResultResponse:
            points = []

            for timestamp, actual, predicted in zip(
                backtest_result.actual.index,
                backtest_result.actual.to_numpy(),
                backtest_result.predicted.to_numpy(),
                strict=True,
            ):
                actual_close = float(actual)
                predicted_close = float(predicted)

                points.append(
                    BacktestPointResponse(
                        timestamp=timestamp.to_pydatetime(),
                        actual_close=actual_close,
                        predicted_close=predicted_close,
                        absolute_error=abs(actual_close - predicted_close),
                    )
                )

            metrics = backtest_result.metrics

            return BacktestModelResultResponse(
                model_name=model_name,
                model_description=description,
                count=len(points),
                metrics=BacktestMetricsResponse(
                    mae=float(metrics["mae"]),
                    rmse=float(metrics["rmse"]),
                    mape=float(metrics["mape"]),
                ),
                data=points,
            )

        naive_response = build_model_response(
            model_name="naive_baseline",
            description=(
                "Repeats the latest available training close for each forecast step."
            ),
            backtest_result=result.naive_result,
        )

        moving_average_response = build_model_response(
            model_name="moving_average_baseline",
            description=(
                "Repeats the average of the latest five available training "
                "closes for each forecast step."
            ),
            backtest_result=result.moving_average_result,
        )

        if naive_response.count != moving_average_response.count:
            raise ValueError("Model evaluation counts do not match.")

        return ModelComparisonResponse(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
            initial_train_size=initial_train_size,
            horizon=horizon,
            count=naive_response.count,
            winner=result.winner,
            naive_baseline=naive_response,
            moving_average_baseline=moving_average_response,
        )

    except (MarketDataError, DataValidationError) as exc:
        logger.warning(
            "Model comparison data failure for %s: %s",
            normalized_symbol,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Market data could not be retrieved or validated.",
        ) from exc

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected model comparison failure for %s",
            normalized_symbol,
        )
        raise HTTPException(
            status_code=502,
            detail="Unable to compare historical forecasts.",
        ) from exc
