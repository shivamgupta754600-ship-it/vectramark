"""Application service for market-specific news retrieval."""

from datetime import UTC, datetime, timedelta
from functools import lru_cache

from vectramark.core.exceptions import ResearchError
from vectramark.data.market_config import get_market
from vectramark.infrastructure.news.rss_news_provider import (
    NewsArticle,
    RSSNewsProvider,
)

MARKET_NEWS_TERMS: dict[str, str] = {
    "XAUUSD": '(gold OR "XAU/USD") (Federal Reserve OR inflation OR dollar OR yields)',
    "XAGUSD": '(silver OR "XAG/USD") (industrial demand OR inflation OR dollar)',
    "NASDAQ100": '("Nasdaq-100" OR Nasdaq) (earnings OR technology OR Federal Reserve)',
    "SP500": '("S&P 500" OR "S&P500") (earnings OR inflation OR Federal Reserve)',
    "DJIA": '("Dow Jones" OR DJIA) (earnings OR economy OR Federal Reserve)',
}


class NewsService:
    """Retrieve, normalize, deduplicate, and filter market news."""

    def __init__(self, provider: RSSNewsProvider | None = None) -> None:
        self.provider = provider or RSSNewsProvider()

    def get_market_news(
        self,
        symbol: str,
        limit: int = 10,
        freshness_hours: int = 72,
    ) -> list[NewsArticle]:
        """Return recent articles and articles with unknown publication dates."""

        normalized_symbol = symbol.strip().upper()

        try:
            get_market(normalized_symbol)
        except (ValueError, KeyError) as exc:
            raise ValueError("Unsupported market symbol.") from exc

        if not 1 <= limit <= 30:
            raise ValueError("Limit must be between 1 and 30.")
        if not 1 <= freshness_hours <= 720:
            raise ValueError("Freshness must be between 1 and 720 hours.")

        query = MARKET_NEWS_TERMS.get(normalized_symbol)
        if query is None:
            raise ValueError("No news search terms are configured for this market.")

        try:
            articles = self.provider.fetch(query=query, limit=limit)
        except ResearchError:
            raise
        except Exception as exc:
            raise ResearchError("News retrieval failed.") from exc

        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=freshness_hours)
        unique: dict[str, NewsArticle] = {}

        for article in articles:
            normalized_url = article.url.strip().rstrip("/")
            if not normalized_url:
                continue

            published_at = article.published_at
            if published_at is not None:
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=UTC)
                published_at = published_at.astimezone(UTC)

                # Exclude stale articles and implausibly future-dated articles.
                if published_at < cutoff or published_at > now + timedelta(hours=1):
                    continue

            key = normalized_url.casefold()
            if key not in unique:
                unique[key] = NewsArticle(
                    title=article.title.strip()[:500],
                    url=normalized_url[:2000],
                    source=article.source.strip()[:200] if article.source else None,
                    published_at=published_at,
                    summary=article.summary.strip()[:2000] if article.summary else None,
                )

        sorted_articles = sorted(
            unique.values(),
            key=lambda item: (
                item.published_at is not None,
                item.published_at or datetime.min.replace(tzinfo=UTC),
            ),
            reverse=True,
        )
        return sorted_articles[:limit]


@lru_cache(maxsize=1)
def get_news_service() -> NewsService:
    """Return the shared news application service."""

    return NewsService()
