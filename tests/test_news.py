"""Tests for market news retrieval and API behavior."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vectramark.api.main import app
from vectramark.application.news_service import NewsService, get_news_service
from vectramark.core.exceptions import ResearchError
from vectramark.infrastructure.news.rss_news_provider import (
    NewsArticle,
    RSSNewsProvider,
)


class FakeNewsProvider:
    """Deterministic provider that avoids live network requests."""

    def __init__(self, articles: list[NewsArticle]) -> None:
        self.articles = articles
        self.last_query: str | None = None
        self.last_limit: int | None = None

    def fetch(self, query: str, limit: int = 20) -> list[NewsArticle]:
        self.last_query = query
        self.last_limit = limit
        return self.articles[:limit]


def make_article(
    title: str,
    url: str,
    published_at: datetime | None = None,
) -> NewsArticle:
    return NewsArticle(
        title=title,
        url=url,
        source="Example News",
        published_at=published_at,
        summary="Example summary",
    )


def test_news_service_returns_recent_articles() -> None:
    recent = datetime.now(UTC) - timedelta(hours=2)
    provider = FakeNewsProvider(
        [make_article("Gold market update", "https://example.com/1", recent)]
    )

    result = NewsService(provider).get_market_news("XAUUSD")

    assert len(result) == 1
    assert result[0].title == "Gold market update"
    assert provider.last_query is not None
    assert "gold" in provider.last_query.lower()


def test_news_service_filters_stale_articles() -> None:
    stale = datetime.now(UTC) - timedelta(hours=100)
    recent = datetime.now(UTC) - timedelta(hours=1)
    provider = FakeNewsProvider(
        [
            make_article("Old headline", "https://example.com/old", stale),
            make_article("Recent headline", "https://example.com/new", recent),
        ]
    )

    result = NewsService(provider).get_market_news(
        "SP500",
        freshness_hours=72,
    )

    assert [article.title for article in result] == ["Recent headline"]


def test_news_service_deduplicates_urls() -> None:
    published = datetime.now(UTC)
    provider = FakeNewsProvider(
        [
            make_article("First", "https://example.com/story/", published),
            make_article("Duplicate", "https://example.com/story", published),
        ]
    )

    result = NewsService(provider).get_market_news("DJIA")

    assert len(result) == 1
    assert result[0].title == "First"


def test_news_service_keeps_articles_with_unknown_dates() -> None:
    provider = FakeNewsProvider(
        [make_article("Undated article", "https://example.com/undated")]
    )

    result = NewsService(provider).get_market_news("XAGUSD")

    assert len(result) == 1
    assert result[0].published_at is None


@pytest.mark.parametrize("symbol", ["UNKNOWN", "", "CRYPTO"])
def test_news_service_rejects_unsupported_symbol(symbol: str) -> None:
    service = NewsService(FakeNewsProvider([]))

    with pytest.raises(ValueError, match="Unsupported market symbol"):
        service.get_market_news(symbol)


@pytest.mark.parametrize(
    ("limit", "freshness_hours"),
    [(0, 72), (31, 72), (10, 0), (10, 721)],
)
def test_news_service_rejects_invalid_parameters(
    limit: int,
    freshness_hours: int,
) -> None:
    service = NewsService(FakeNewsProvider([]))

    with pytest.raises(ValueError):
        service.get_market_news(
            "XAUUSD",
            limit=limit,
            freshness_hours=freshness_hours,
        )


def test_rss_provider_rejects_invalid_query_and_limit() -> None:
    provider = RSSNewsProvider()

    with pytest.raises(ValueError):
        provider.fetch(" ")

    with pytest.raises(ValueError):
        provider.fetch("gold", limit=31)


def test_rss_provider_parses_dates() -> None:
    parsed = RSSNewsProvider._parse_date("Mon, 05 Jan 2026 12:00:00 GMT")

    assert parsed == datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def test_rss_provider_returns_none_for_invalid_dates() -> None:
    assert RSSNewsProvider._parse_date("not a date") is None
    assert RSSNewsProvider._parse_date(None) is None


def test_news_endpoint_success(monkeypatch: pytest.MonkeyPatch) -> None:
    article = make_article(
        "Gold market update",
        "https://example.com/gold",
        datetime.now(UTC) - timedelta(hours=1),
    )

    class FakeService:
        def get_market_news(
            self,
            symbol: str,
            limit: int = 10,
            freshness_hours: int = 72,
        ) -> list[NewsArticle]:
            assert symbol == "XAUUSD"
            assert limit == 5
            assert freshness_hours == 24
            return [article]

    app.dependency_overrides[get_news_service] = lambda: FakeService()

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/analytics/news/xauusd",
                params={"limit": 5, "freshness_hours": 24},
            )
    finally:
        app.dependency_overrides.pop(get_news_service, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XAUUSD"
    assert payload["count"] == 1
    assert payload["data"][0]["title"] == "Gold market update"
    assert payload["data"][0]["publication_time_status"] == "known"
    assert "not independently verified" in payload["disclaimer"]


def test_news_endpoint_rejects_unknown_symbol() -> None:
    class FakeService:
        def get_market_news(self, symbol: str, **kwargs: Any) -> list[NewsArticle]:
            raise ValueError("Unsupported market symbol.")

    app.dependency_overrides[get_news_service] = lambda: FakeService()

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/analytics/news/UNKNOWN")
    finally:
        app.dependency_overrides.pop(get_news_service, None)

    assert response.status_code == 404


def test_news_endpoint_maps_provider_failure() -> None:
    class FakeService:
        def get_market_news(self, symbol: str, **kwargs: Any) -> list[NewsArticle]:
            raise ResearchError("Provider failed.")

    app.dependency_overrides[get_news_service] = lambda: FakeService()

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/analytics/news/XAUUSD")
    finally:
        app.dependency_overrides.pop(get_news_service, None)

    assert response.status_code == 502


def test_news_endpoint_validates_query_parameters() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/analytics/news/XAUUSD",
            params={"limit": 100},
        )

    assert response.status_code == 422
