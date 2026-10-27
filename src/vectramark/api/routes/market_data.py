import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from vectramark.api.dependencies import (
    get_market_data_quality_service,
    get_market_data_service,
)
from vectramark.api.schemas.market_data import (
    HistoricalMarketDataResponse,
    MarketCandleResponse,
    MarketInstrumentMetadata,
    MarketInterval,
    MarketPeriod,
)
from vectramark.api.schemas.market_data_quality import MarketDataQualityResponse
from vectramark.application.market_data_quality_service import (
    MarketDataQualityService,
)
from vectramark.application.market_data_service import MarketDataService
from vectramark.core.exceptions import DataValidationError, MarketDataError
from vectramark.data.market_config import get_market
from vectramark.infrastructure.market_data.symbol_mapping import (
    DATA_PROVIDER_NAME,
    get_provider_instrument,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/market-data",
    tags=["Market Data"],
)


def _normalize_symbol(symbol: str) -> str:
    """Normalize a market symbol or reject an empty value."""
    normalized = symbol.strip().upper()

    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="Market symbol cannot be empty.",
        )

    return normalized


def build_instrument_metadata(symbol: str) -> MarketInstrumentMetadata:
    """Build API metadata from the configured market and provider mapping."""
    market = get_market(symbol)
    provider_instrument = get_provider_instrument(symbol)

    return MarketInstrumentMetadata(
        internal_symbol=market.symbol,
        name=market.name,
        instrument_type=provider_instrument.instrument_type,
        provider_name=DATA_PROVIDER_NAME,
        provider_symbol=provider_instrument.provider_symbol,
        description=provider_instrument.description,
    )


def _raise_market_data_http_error(
    symbol: str,
    exc: Exception,
) -> None:
    """Log a provider/validation failure without exposing internals."""
    logger.warning(
        "Market data request failed for symbol %s: %s",
        symbol,
        type(exc).__name__,
    )

    raise HTTPException(
        status_code=502,
        detail="Market data could not be retrieved or validated.",
    ) from exc


@router.get(
    "/history/{symbol}",
    response_model=HistoricalMarketDataResponse,
    summary="Get historical market data",
    description=(
        "Fetch validated historical market data, save it locally, and identify "
        "the configured instrument and provider ticker in the response."
    ),
)
def get_historical_market_data(
    symbol: str,
    period: MarketPeriod = Query(default="1y"),
    interval: MarketInterval = Query(default="1d"),
    service: MarketDataService = Depends(get_market_data_service),
) -> HistoricalMarketDataResponse:
    """Fetch validated historical market data for a supported market."""
    normalized_symbol = _normalize_symbol(symbol)

    try:
        market_data = service.fetch_and_store(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Unsupported market symbol.",
        ) from exc
    except (MarketDataError, DataValidationError) as exc:
        _raise_market_data_http_error(normalized_symbol, exc)

    candles: list[MarketCandleResponse] = []

    for timestamp, row in market_data.iterrows():
        raw_volume = row.get("Volume")
        volume = (
            None if raw_volume is None or pd.isna(raw_volume) else float(raw_volume)
        )

        candles.append(
            MarketCandleResponse(
                timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=volume,
            )
        )

    try:
        instrument = build_instrument_metadata(normalized_symbol)
    except (ValueError, MarketDataError) as exc:
        _raise_market_data_http_error(normalized_symbol, exc)

    return HistoricalMarketDataResponse(
        symbol=normalized_symbol,
        period=period,
        interval=interval,
        count=len(candles),
        data=candles,
        instrument=instrument,
    )


@router.get(
    "/quality/{symbol}",
    response_model=MarketDataQualityResponse,
    summary="Assess historical market data quality",
    description=(
        "Check OHLC integrity, duplicate timestamps, history sufficiency, "
        "and freshness of the latest candle."
    ),
)
def get_market_data_quality(
    symbol: str,
    period: MarketPeriod = Query(default="1y"),
    interval: MarketInterval = Query(default="1d"),
    service: MarketDataQualityService = Depends(get_market_data_quality_service),
) -> MarketDataQualityResponse:
    """Retrieve validated data and return its quality assessment."""
    normalized_symbol = _normalize_symbol(symbol)

    try:
        assessment = service.assess(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=404,
            detail="Unsupported market symbol.",
        ) from exc
    except (MarketDataError, DataValidationError) as exc:
        _raise_market_data_http_error(normalized_symbol, exc)

    return MarketDataQualityResponse(
        symbol=normalized_symbol,
        period=period,
        interval=interval,
        **assessment,
    )


@router.get(
    "/supported-symbols",
    response_model=list[str],
    summary="List supported market symbols",
)
def get_supported_market_symbols() -> list[str]:
    """Return the internal symbols supported by VectraMark."""
    return [
        "XAUUSD",
        "XAGUSD",
        "NASDAQ100",
        "SP500",
        "DJIA",
    ]
