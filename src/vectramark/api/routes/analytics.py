"""API routes for historical financial analytics and market bias."""

import logging
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from vectramark.api.analytics_dependencies import (
    get_analytics_service,
    get_market_research_snapshot_service,
)
from vectramark.api.dependencies import get_market_data_service
from vectramark.api.schemas.analytics import (
    AnalyticsResponse,
    BiasSignalResponse,
    MarketBiasResponse,
)
from vectramark.api.schemas.market_data import (
    MarketInterval,
    MarketPeriod,
    MarketSymbol,
)
from vectramark.api.schemas.market_safety import MarketSafetyResponse
from vectramark.api.schemas.news import (
    MarketNewsResponse,
    NewsArticleResponse,
)
from vectramark.api.schemas.news_context import (
    MarketNewsContextResponse,
    NewsContextArticleResponse,
)
from vectramark.api.schemas.research_snapshot import (
    MarketResearchSnapshotResponse,
)
from vectramark.api.schemas.support_resistance import (
    SupportResistanceLevelResponse,
    SupportResistanceResponse,
)
from vectramark.application.market_bias_service import MarketBiasService
from vectramark.application.market_data_service import MarketDataService
from vectramark.application.market_research_snapshot_service import (
    MarketResearchSnapshotService,
)
from vectramark.application.market_safety_service import MarketSafetyService
from vectramark.application.news_context_service import (
    NewsContextService,
)
from vectramark.application.news_service import NewsService, get_news_service
from vectramark.application.support_resistance_service import (
    SupportResistanceService,
)
from vectramark.core.exceptions import (
    DataValidationError,
    MarketDataError,
    ResearchError,
)
from vectramark.data.market_config import get_market
from vectramark.domain.analytics.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Analytics"],
)


@router.get(
    "/history/{symbol}",
    response_model=AnalyticsResponse,
)
def get_historical_analytics(
    symbol: str,
    period: MarketPeriod = Query(default="1y"),
    interval: MarketInterval = Query(default="1d"),
    sma_window: int = Query(default=20, ge=1, le=250),
    ema_window: int = Query(default=20, ge=1, le=250),
    rsi_window: int = Query(default=14, ge=1, le=100),
    volatility_window: int = Query(default=20, ge=2, le=250),
    periods_per_year: int = Query(default=252, ge=1, le=366),
    market_service: MarketDataService = Depends(get_market_data_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsResponse:
    """Return historical prices enriched with financial analytics."""

    normalized_symbol = symbol.strip().upper()

    try:
        market_data = market_service.fetch_and_store(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
        )

        if not isinstance(market_data, pd.DataFrame):
            raise ValueError("Market-data service returned invalid data.")

        if "Close" not in market_data.columns:
            raise ValueError("Market data must contain a Close column.")

        result = analytics_service.analyze(
            market_data,
            sma_window=sma_window,
            ema_window=ema_window,
            rsi_window=rsi_window,
            volatility_window=volatility_window,
            periods_per_year=periods_per_year,
        )

        result = result.replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )

        records = result.reset_index().to_dict(orient="records")
        data = []

        for record in records:
            safe_record = {}

            for key, value in record.items():
                if pd.isna(value):
                    safe_record[key] = None
                elif isinstance(value, pd.Timestamp):
                    safe_record[key] = value.isoformat()
                elif hasattr(value, "item"):
                    safe_record[key] = value.item()
                else:
                    safe_record[key] = value

            data.append(safe_record)

        return AnalyticsResponse(
            symbol=normalized_symbol,
            count=len(data),
            sma_window=sma_window,
            ema_window=ema_window,
            rsi_window=rsi_window,
            volatility_window=volatility_window,
            periods_per_year=periods_per_year,
            data=data,
        )

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to calculate analytics for symbol %s",
            normalized_symbol,
        )

        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve or calculate market analytics.",
        ) from exc


@router.get(
    "/bias/{symbol}",
    response_model=MarketBiasResponse,
    summary="Estimate market bias from technical indicators",
    description=(
        "Return an explainable bullish, bearish, or neutral technical bias. "
        "Analysis uses daily closing prices and rule-based signals. "
        "This is not a guarantee of future market direction."
    ),
)
def get_market_bias(
    symbol: str,
    period: MarketPeriod = Query(default="5y"),
    timeframe: Literal["daily", "weekly", "monthly", "yearly"] = Query(default="daily"),
    market_service: MarketDataService = Depends(get_market_data_service),
) -> MarketBiasResponse:
    """Return technical signals and their combined market-bias score."""

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
        result = MarketBiasService(market_service).analyze(
            symbol=normalized_symbol,
            period=period,
            timeframe=timeframe,
        )

        return MarketBiasResponse(
            symbol=normalized_symbol,
            period=period,
            timeframe=result.timeframe,
            as_of=result.as_of.to_pydatetime(),
            latest_close=result.latest_close,
            observations=result.observations,
            bias=result.bias,
            score=result.score,
            explanation=result.explanation,
            signals=[
                BiasSignalResponse(
                    name=signal.name,
                    value=signal.value,
                    unit=signal.unit,
                    direction=signal.direction,
                    explanation=signal.explanation,
                )
                for signal in result.signals
            ],
            disclaimer=(
                "This is a rule-based technical analysis, not financial advice "
                "or a guarantee of future prices. Consider uncertainty, "
                "transaction costs, and other market information."
            ),
        )

    except (MarketDataError, DataValidationError) as exc:
        logger.warning(
            "Market-bias data failure for %s: %s",
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
            "Unexpected market-bias failure for %s",
            normalized_symbol,
        )
        raise HTTPException(
            status_code=502,
            detail="Unable to calculate market bias.",
        ) from exc


@router.get(
    "/levels/{symbol}",
    response_model=SupportResistanceResponse,
    summary="Detect historical support and resistance zones",
    description=(
        "Identify historical swing highs and lows and cluster nearby prices "
        "into estimated zones. These zones do not guarantee future reactions "
        "and are not financial advice."
    ),
)
def get_support_resistance(
    symbol: str,
    period: MarketPeriod = Query(default="1y"),
    swing_window: int = Query(default=3, ge=1, le=20),
    tolerance_pct: float = Query(default=0.5, ge=0, le=5),
    max_levels: int = Query(default=10, ge=1, le=50),
    market_service: MarketDataService = Depends(get_market_data_service),
) -> SupportResistanceResponse:
    """Return estimated support and resistance zones for a market."""

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
        result = SupportResistanceService(market_service).analyze(
            symbol=normalized_symbol,
            period=period,
            swing_window=swing_window,
            tolerance_pct=tolerance_pct,
            max_levels=max_levels,
        )

        return SupportResistanceResponse(
            symbol=normalized_symbol,
            period=period,
            as_of=pd.Timestamp.now(tz="UTC").to_pydatetime(),
            latest_close=result.latest_close,
            observations=result.observations,
            swing_window=swing_window,
            tolerance_pct=tolerance_pct,
            levels=[
                SupportResistanceLevelResponse(
                    price=level.price,
                    zone_low=level.zone_low,
                    zone_high=level.zone_high,
                    kind=level.kind,
                    touch_count=level.touch_count,
                    explanation=level.explanation,
                )
                for level in result.levels
            ],
            disclaimer=(
                "Support and resistance zones are estimates based on "
                "historical prices, not guaranteed turning points or "
                "financial advice. Market conditions can change."
            ),
        )

    except (MarketDataError, DataValidationError) as exc:
        logger.warning(
            "Support/resistance data failure for %s: %s",
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
            "Unexpected support/resistance failure for %s",
            normalized_symbol,
        )
        raise HTTPException(
            status_code=502,
            detail="Unable to calculate support and resistance zones.",
        ) from exc


@router.get(
    "/news/{symbol}",
    response_model=MarketNewsResponse,
    summary="Retrieve recent news relevant to a supported market",
    description=(
        "Retrieve headlines from an RSS news provider. Headlines are not "
        "verified facts and must not be treated as predictions or trading advice."
    ),
)
def get_market_news(
    symbol: str,
    limit: int = Query(default=10, ge=1, le=30),
    freshness_hours: int = Query(default=72, ge=1, le=720),
    news_service: NewsService = Depends(get_news_service),
) -> MarketNewsResponse:
    """Return recent market-related headlines."""

    normalized_symbol = symbol.strip().upper()

    if not normalized_symbol:
        raise HTTPException(
            status_code=400,
            detail="Market symbol cannot be empty.",
        )

    try:
        articles = news_service.get_market_news(
            symbol=normalized_symbol,
            limit=limit,
            freshness_hours=freshness_hours,
        )
    except ValueError as exc:
        if "Unsupported market symbol" in str(exc):
            raise HTTPException(
                status_code=404,
                detail="Unsupported market symbol.",
            ) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ResearchError as exc:
        logger.warning(
            "News retrieval failed for %s: %s",
            normalized_symbol,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="News provider is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected news failure for %s", normalized_symbol)
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve market news.",
        ) from exc

    return MarketNewsResponse(
        symbol=normalized_symbol,
        count=len(articles),
        freshness_hours=freshness_hours,
        data=[
            NewsArticleResponse(
                title=article.title,
                url=article.url,
                source=article.source,
                published_at=article.published_at,
                summary=article.summary,
                publication_time_status=(
                    "known" if article.published_at is not None else "unknown"
                ),
            )
            for article in articles
        ],
        disclaimer=(
            "These are third-party reported headlines, not independently "
            "verified facts. Missing publication dates cannot be freshness-"
            "verified. News alone does not predict price direction. This "
            "information is not financial advice."
        ),
    )


@router.get(
    "/news/{symbol}/context",
    response_model=MarketNewsContextResponse,
    summary="Explain themes in recent market news",
    description=(
        "Classify news headlines using transparent phrase-matching rules. "
        "Confidence measures rule-match strength, not factual certainty "
        "or the likelihood of a price move."
    ),
)
def get_market_news_context(
    symbol: str,
    limit: int = Query(default=10, ge=1, le=30),
    freshness_hours: int = Query(default=72, ge=1, le=720),
    news_service: NewsService = Depends(get_news_service),
) -> MarketNewsContextResponse:
    """Return recent headlines with explainable context labels."""

    normalized_symbol = symbol.strip().upper()

    if not normalized_symbol:
        raise HTTPException(
            status_code=400,
            detail="Market symbol cannot be empty.",
        )

    try:
        context_service = NewsContextService(news_service)
        results = context_service.analyze_market_news(
            symbol=normalized_symbol,
            limit=limit,
            freshness_hours=freshness_hours,
        )
    except ValueError as exc:
        if "Unsupported market symbol" in str(exc):
            raise HTTPException(
                status_code=404,
                detail="Unsupported market symbol.",
            ) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ResearchError as exc:
        logger.warning(
            "News context retrieval failed for %s: %s",
            normalized_symbol,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="News provider is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected news context failure for %s",
            normalized_symbol,
        )
        raise HTTPException(
            status_code=502,
            detail="Unable to analyze market news.",
        ) from exc

    return MarketNewsContextResponse(
        symbol=normalized_symbol,
        count=len(results),
        freshness_hours=freshness_hours,
        data=[
            NewsContextArticleResponse(
                title=item.article.title,
                url=item.article.url,
                source=item.article.source,
                published_at=item.article.published_at,
                publication_time_status=(
                    "known" if item.article.published_at is not None else "unknown"
                ),
                theme=item.context.theme,
                confidence=item.context.confidence,
                evidence=list(item.context.evidence),
                market_relevance=item.context.market_relevance,
                interpretation=item.context.interpretation,
            )
            for item in results
        ],
        confidence_note=(
            "Confidence reflects the number of matching phrases in configured "
            "rules. It is not a calibrated probability, source reliability "
            "score, or prediction of market direction."
        ),
        disclaimer=(
            "Headlines are third-party content and have not been independently "
            "verified. Context labels are rule-based interpretations, not "
            "financial advice or buy/sell signals."
        ),
    )


@router.get(
    "/news/{symbol}/safety",
    response_model=MarketSafetyResponse,
    summary="Check limitations and risk flags in recent market headlines",
    description=(
        "Flags uncertain timestamps, weak rule matches, promotional language, "
        "and mixed directional wording. This is not a market forecast."
    ),
)
def get_market_news_safety(
    symbol: str,
    limit: int = Query(default=10, ge=1, le=30),
    freshness_hours: int = Query(default=72, ge=1, le=720),
    news_service: NewsService = Depends(get_news_service),
) -> MarketSafetyResponse:
    """Return conservative safety flags for recent market-news context."""

    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise HTTPException(
            status_code=400,
            detail="Market symbol cannot be empty.",
        )

    try:
        result = MarketSafetyService(NewsContextService(news_service)).assess(
            symbol=normalized_symbol,
            limit=limit,
            freshness_hours=freshness_hours,
        )
        assessment = result.assessment
        return MarketSafetyResponse(
            symbol=normalized_symbol,
            status=assessment.status,
            review_required=assessment.review_required,
            flags=list(assessment.flags),
            checked_articles=assessment.checked_articles,
            disclaimer=assessment.disclaimer,
        )
    except ValueError as exc:
        if "Unsupported market symbol" in str(exc):
            raise HTTPException(
                status_code=404,
                detail="Unsupported market symbol.",
            ) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ResearchError as exc:
        logger.warning(
            "News safety assessment failed for %s: %s",
            normalized_symbol,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="News provider is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected news safety failure for %s",
            normalized_symbol,
        )
        raise HTTPException(
            status_code=502,
            detail="Unable to assess market-news safety.",
        ) from exc


@router.get(
    "/snapshot/{symbol}",
    response_model=MarketResearchSnapshotResponse,
    summary="Get a combined market research snapshot",
)
def get_market_research_snapshot(
    symbol: MarketSymbol,
    period: MarketPeriod = "1y",
    interval: Literal["1d", "1wk", "1mo"] = "1d",
    timeframe: Literal["daily", "weekly", "monthly", "yearly"] = "daily",
    news_limit: int = Query(default=10, ge=1, le=20),
    freshness_hours: int = Query(default=72, ge=1, le=720),
    service: MarketResearchSnapshotService = Depends(
        get_market_research_snapshot_service
    ),
) -> dict:
    """Combine five analysis sections without hiding partial failures."""
    result = service.create_snapshot(
        symbol=symbol,
        period=period,
        interval=interval,
        timeframe=timeframe,
        news_limit=news_limit,
        freshness_hours=freshness_hours,
    )
    return jsonable_encoder(result)
