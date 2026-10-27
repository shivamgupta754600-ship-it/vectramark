"""Tests for the combined market research snapshot service."""

from unittest.mock import Mock

from vectramark.application.market_research_snapshot_service import (
    MarketResearchSnapshotService,
)


def make_service() -> MarketResearchSnapshotService:
    """Create a snapshot service with predictable mock dependencies."""
    bias = Mock()
    bias.analyze.return_value = {"bias": "neutral", "score": 0}

    support_resistance = Mock()
    support_resistance.analyze.return_value = {"levels": []}

    news_context = Mock()
    news_context.analyze_market_news.return_value = {"count": 2, "data": []}

    market_safety = Mock()
    market_safety.assess.return_value = {
        "status": "no_flags_detected",
        "review_required": False,
    }

    data_quality = Mock()
    data_quality.assess.return_value = {"status": "acceptable"}

    return MarketResearchSnapshotService(
        bias_service=bias,
        support_resistance_service=support_resistance,
        news_context_service=news_context,
        market_safety_service=market_safety,
        market_data_quality_service=data_quality,
    )


def test_snapshot_contains_all_five_sections() -> None:
    service = make_service()

    result = service.create_snapshot(symbol="XAUUSD")

    assert result["symbol"] == "XAUUSD"
    assert result["period"] == "1y"
    assert result["interval"] == "1d"

    expected_sections = {
        "bias",
        "support_resistance",
        "news_context",
        "news_safety",
        "data_quality",
    }
    assert expected_sections.issubset(result.keys())

    for section_name in expected_sections:
        section = result[section_name]
        assert section["status"] == "ok"
        assert section["generated_at"] is not None
        assert section["data"] is not None
        assert section["error_code"] is None


def test_one_section_failure_does_not_erase_other_sections() -> None:
    service = make_service()
    service._news_context_service.analyze_market_news.side_effect = RuntimeError(
        "provider failure"
    )

    result = service.create_snapshot(symbol="XAUUSD")

    assert result["news_context"]["status"] == "error"
    assert result["news_context"]["data"] is None
    assert result["news_context"]["error_code"] == "RuntimeError"
    assert "provider failure" not in result["news_context"]["message"]

    assert result["bias"]["status"] == "ok"
    assert result["support_resistance"]["status"] == "ok"
    assert result["news_safety"]["status"] == "ok"
    assert result["data_quality"]["status"] == "ok"


def test_snapshot_passes_query_parameters_to_services() -> None:
    service = make_service()

    service.create_snapshot(
        symbol="XAUUSD",
        period="3mo",
        interval="1wk",
        timeframe="weekly",
        news_limit=5,
        freshness_hours=48,
    )

    service._bias_service.analyze.assert_called_once_with(
        "XAUUSD",
        "3mo",
        "weekly",
    )
    service._support_resistance_service.analyze.assert_called_once_with(
        "XAUUSD",
        "3mo",
    )
    service._news_context_service.analyze_market_news.assert_called_once_with(
        "XAUUSD",
        limit=5,
        freshness_hours=48,
    )
    service._market_safety_service.assess.assert_called_once_with(
        "XAUUSD",
        limit=5,
        freshness_hours=48,
    )
    service._market_data_quality_service.assess.assert_called_once_with(
        "XAUUSD",
        period="3mo",
        interval="1wk",
    )


def test_all_sections_can_report_failure_without_crashing() -> None:
    service = make_service()

    service._bias_service.analyze.side_effect = ValueError("invalid data")
    service._support_resistance_service.analyze.side_effect = ValueError("invalid data")
    service._news_context_service.analyze_market_news.side_effect = RuntimeError(
        "news provider unavailable"
    )
    service._market_safety_service.assess.side_effect = RuntimeError(
        "safety analysis unavailable"
    )
    service._market_data_quality_service.assess.side_effect = RuntimeError(
        "data quality unavailable"
    )

    result = service.create_snapshot(symbol="XAUUSD")

    for section_name in (
        "bias",
        "support_resistance",
        "news_context",
        "news_safety",
        "data_quality",
    ):
        assert result[section_name]["status"] == "error"
        assert result[section_name]["data"] is None
        assert result[section_name]["message"]
