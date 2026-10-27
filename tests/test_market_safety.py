"""Tests for conservative market-news safety checks."""

from vectramark.domain.analytics.market_safety import assess_news_safety


def test_empty_articles_have_no_flags() -> None:
    result = assess_news_safety([])

    assert result.checked_articles == 0
    assert result.flags == ()
    assert result.review_required is False
    assert result.status == "no_flags_detected"


def test_unknown_publication_time_requires_review() -> None:
    result = assess_news_safety(
        [
            {
                "title": "Central bank discusses interest rates",
                "theme": "interest_rates",
                "confidence": "medium",
                "publication_time_status": "unknown",
            }
        ]
    )

    assert result.review_required is True
    assert "publication_time_unknown" in result.flags


def test_uncertain_low_confidence_headline_is_flagged() -> None:
    result = assess_news_safety(
        [
            {
                "title": "Markets update",
                "theme": "uncertain",
                "confidence": "low",
                "publication_time_status": "known",
            }
        ]
    )

    assert "unclassified_headline" in result.flags
    assert "low_rule_match_confidence" in result.flags


def test_mixed_directional_language_is_flagged() -> None:
    result = assess_news_safety(
        [
            {
                "title": "Gold rally continues",
                "theme": "uncertain",
                "confidence": "low",
                "publication_time_status": "known",
            },
            {
                "title": "Gold plunges amid market pressure",
                "theme": "uncertain",
                "confidence": "low",
                "publication_time_status": "known",
            },
        ]
    )

    assert "mixed_directional_headline_language" in result.flags


def test_guaranteed_profit_language_is_flagged() -> None:
    result = assess_news_safety(
        [
            {
                "title": "Guaranteed profit from this market strategy",
                "theme": "corporate_earnings",
                "confidence": "high",
                "publication_time_status": "known",
            }
        ]
    )

    assert "promotional_or_guarantee_language" in result.flags


def test_no_flags_does_not_claim_a_trade_is_safe() -> None:
    result = assess_news_safety(
        [
            {
                "title": "Federal Reserve discusses monetary policy",
                "theme": "interest_rates",
                "confidence": "high",
                "publication_time_status": "known",
            }
        ]
    )

    assert result.status == "no_flags_detected"
    assert "does not mean a trade is safe" in result.disclaimer
