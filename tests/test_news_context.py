"""Tests for rule-based market-news context analysis."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from vectramark.api.main import app
from vectramark.application.news_service import get_news_service
from vectramark.domain.analytics.news_context import analyze_headline
from vectramark.infrastructure.news.rss_news_provider import NewsArticle


def test_classifies_interest_rate_headline() -> None:
    result = analyze_headline(
        "XAUUSD",
        "Federal Reserve signals possible interest rate cut",
    )

    assert result.theme == "interest_rates"
    assert "federal reserve" in result.evidence
    assert result.confidence == "high"
    assert "gold" in result.market_relevance.lower()


def test_classifies_silver_industrial_demand() -> None:
    result = analyze_headline(
        "XAGUSD",
        "Silver demand rises as solar demand increases",
    )

    assert result.theme == "industrial_demand"
    assert len(result.evidence) >= 2
    assert result.confidence == "medium"


def test_classifies_corporate_earnings_for_sp500() -> None:
    result = analyze_headline(
        "SP500",
        "Quarterly results exceed expectations",
    )

    assert result.theme == "corporate_earnings"
    assert result.confidence == "low"
    assert "earnings" in result.market_relevance.lower()


def test_unknown_headline_requires_manual_review() -> None:
    result = analyze_headline(
        "XAUUSD",
        "A company announces a new office location",
    )

    assert result.theme == "uncertain"
    assert result.confidence == "low"
    assert result.evidence == ()


def test_summary_can_provide_matching_evidence() -> None:
    result = analyze_headline(
        "NASDAQ100",
        "Markets await company results",
        summary="Technology earnings outlook remains in focus.",
    )

    assert result.theme == "corporate_earnings"
    assert "earnings" in result.evidence


def test_classification_is_case_insensitive() -> None:
    result = analyze_headline(
        "DJIA",
        "QUARTERLY RESULTS SHOW HIGHER EARNINGS",
    )

    assert result.theme == "corporate_earnings"


def test_context_endpoint_returns_explanations() -> None:
    article = NewsArticle(
        title="Federal Reserve signals possible interest rate cut",
        url="https://example.com/rates",
        source="Example News",
        published_at=datetime.now(UTC),
        summary="The central bank discussed monetary policy.",
    )

    class FakeNewsService:
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

    app.dependency_overrides[get_news_service] = lambda: FakeNewsService()

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/analytics/news/XAUUSD/context",
                params={"limit": 5, "freshness_hours": 24},
            )
    finally:
        app.dependency_overrides.pop(get_news_service, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XAUUSD"
    assert payload["count"] == 1
    assert payload["data"][0]["theme"] == "interest_rates"
    assert payload["data"][0]["evidence"]
    assert payload["data"][0]["confidence"] in {"low", "medium", "high"}
    assert "not a calibrated probability" in payload["confidence_note"]


def test_context_endpoint_rejects_invalid_limit() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/analytics/news/XAUUSD/context",
            params={"limit": 100},
        )

    assert response.status_code == 422


def test_context_endpoint_maps_provider_failure() -> None:
    from vectramark.core.exceptions import ResearchError

    class FakeNewsService:
        def get_market_news(
            self,
            symbol: str,
            limit: int = 10,
            freshness_hours: int = 72,
        ) -> list[NewsArticle]:
            raise ResearchError("Provider unavailable.")

    app.dependency_overrides[get_news_service] = lambda: FakeNewsService()

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/analytics/news/XAUUSD/context")
    finally:
        app.dependency_overrides.pop(get_news_service, None)

    assert response.status_code == 502
