"""Application service for explaining the context of market news."""

from dataclasses import dataclass

from vectramark.application.news_service import NewsService, get_news_service
from vectramark.core.exceptions import ResearchError
from vectramark.domain.analytics.news_context import (
    NewsContext,
    analyze_headline,
)
from vectramark.infrastructure.news.rss_news_provider import NewsArticle


@dataclass(frozen=True)
class AnalyzedNewsArticle:
    """A news article paired with its explainable context."""

    article: NewsArticle
    context: NewsContext


class NewsContextService:
    """Retrieve market news and classify each article."""

    def __init__(self, news_service: NewsService | None = None) -> None:
        self.news_service = news_service or get_news_service()

    def analyze_market_news(
        self,
        symbol: str,
        limit: int = 10,
        freshness_hours: int = 72,
    ) -> list[AnalyzedNewsArticle]:
        """Return articles with rule-based context analysis."""

        normalized_symbol = symbol.strip().upper()

        try:
            articles = self.news_service.get_market_news(
                symbol=normalized_symbol,
                limit=limit,
                freshness_hours=freshness_hours,
            )
        except ResearchError:
            raise
        except ValueError:
            raise
        except Exception as exc:
            raise ResearchError("News context retrieval failed.") from exc

        results: list[AnalyzedNewsArticle] = []

        for article in articles:
            context = analyze_headline(
                symbol=normalized_symbol,
                title=article.title,
                summary=article.summary,
            )
            results.append(
                AnalyzedNewsArticle(
                    article=article,
                    context=context,
                )
            )

        return results
