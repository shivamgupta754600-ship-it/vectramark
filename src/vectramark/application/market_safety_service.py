"""Application service for market-news safety checks."""

from dataclasses import dataclass

from vectramark.application.news_context_service import NewsContextService
from vectramark.domain.analytics.market_safety import (
    NewsSafetyAssessment,
    assess_news_safety,
)


@dataclass(frozen=True)
class MarketSafetyResult:
    """News context plus its conservative safety assessment."""

    assessment: NewsSafetyAssessment


class MarketSafetyService:
    """Run existing news analysis before checking its limitations."""

    def __init__(self, context_service: NewsContextService | None = None) -> None:
        self.context_service = context_service or NewsContextService()

    def assess(
        self,
        symbol: str,
        limit: int = 10,
        freshness_hours: int = 72,
    ) -> MarketSafetyResult:
        """Retrieve news and flag uncertainty or risky headline language."""

        analyzed = self.context_service.analyze_market_news(
            symbol=symbol,
            limit=limit,
            freshness_hours=freshness_hours,
        )

        articles = [
            {
                "title": item.article.title,
                "theme": item.context.theme,
                "confidence": item.context.confidence,
                "publication_time_status": (
                    "known" if item.article.published_at is not None else "unknown"
                ),
            }
            for item in analyzed
        ]

        return MarketSafetyResult(
            assessment=assess_news_safety(articles),
        )
