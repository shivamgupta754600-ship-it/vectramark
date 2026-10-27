"""Orchestrate independent analyses into one market research snapshot."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

SNAPSHOT_DISCLAIMER = (
    "For research and educational purposes only. This snapshot is not "
    "financial advice or a recommendation to buy or sell an asset. "
    "Market data, news interpretations, and analytical results may be "
    "incomplete, delayed, or incorrect. Verify information independently."
)


class MarketResearchSnapshotService:
    """Collect market-analysis results while isolating section failures."""

    def __init__(
        self,
        bias_service: Any,
        support_resistance_service: Any,
        news_context_service: Any,
        market_safety_service: Any,
        market_data_quality_service: Any,
    ) -> None:
        self._bias_service = bias_service
        self._support_resistance_service = support_resistance_service
        self._news_context_service = news_context_service
        self._market_safety_service = market_safety_service
        self._market_data_quality_service = market_data_quality_service

    @staticmethod
    def _run_section(
        section_name: str,
        symbol: str,
        operation: Callable[[], Any],
    ) -> dict[str, Any]:
        """Run one section and represent success or failure consistently."""
        generated_at = datetime.now(UTC)

        try:
            result = operation()
            return {
                "status": "ok",
                "generated_at": generated_at,
                "data": result,
                "error_code": None,
                "message": None,
            }
        except Exception as exc:
            logger.exception(
                "Market research snapshot section failed",
                extra={
                    "section": section_name,
                    "symbol": symbol,
                    "error_type": type(exc).__name__,
                },
            )
            return {
                "status": "error",
                "generated_at": generated_at,
                "data": None,
                "error_code": type(exc).__name__,
                "message": (
                    "This analysis section is temporarily unavailable. "
                    "Check application logs for diagnostic details."
                ),
            }

    def create_snapshot(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        timeframe: str = "daily",
        news_limit: int = 10,
        freshness_hours: int = 72,
    ) -> dict[str, Any]:
        """Return a snapshot without letting one section erase others."""
        requested_at = datetime.now(UTC)

        bias = self._run_section(
            "bias",
            symbol,
            lambda: self._bias_service.analyze(
                symbol,
                period,
                timeframe,
            ),
        )

        support_resistance = self._run_section(
            "support_resistance",
            symbol,
            lambda: self._support_resistance_service.analyze(
                symbol,
                period,
            ),
        )

        news_context = self._run_section(
            "news_context",
            symbol,
            lambda: self._news_context_service.analyze_market_news(
                symbol,
                limit=news_limit,
                freshness_hours=freshness_hours,
            ),
        )

        news_safety = self._run_section(
            "news_safety",
            symbol,
            lambda: self._market_safety_service.assess(
                symbol,
                limit=news_limit,
                freshness_hours=freshness_hours,
            ),
        )

        data_quality = self._run_section(
            "data_quality",
            symbol,
            lambda: self._market_data_quality_service.assess(
                symbol,
                period=period,
                interval=interval,
            ),
        )

        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "timeframe": timeframe,
            "requested_at": requested_at,
            "bias": bias,
            "support_resistance": support_resistance,
            "news_context": news_context,
            "news_safety": news_safety,
            "data_quality": data_quality,
            "disclaimer": SNAPSHOT_DISCLAIMER,
        }
