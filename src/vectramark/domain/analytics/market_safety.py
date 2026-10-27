"""Conservative safety checks for rule-based market-news context."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NewsSafetyAssessment:
    """Safety metadata; this does not establish truth or predict prices."""

    status: str
    review_required: bool
    flags: tuple[str, ...]
    checked_articles: int
    disclaimer: str


BULLISH_TERMS = (
    "bullish",
    "surge",
    "surges",
    "rally",
    "rallies",
    "soars",
    "jumped",
    "gains",
)
BEARISH_TERMS = (
    "bearish",
    "crash",
    "crashes",
    "plunge",
    "plunges",
    "slumps",
    "tumbles",
    "falls sharply",
)
PROMOTIONAL_TERMS = (
    "guaranteed profit",
    "risk-free",
    "100% accurate",
    "sure win",
    "get rich quick",
)


def assess_news_safety(articles: list[dict[str, object]]) -> NewsSafetyAssessment:
    """Flag uncertainty and risky language without giving trading advice."""

    flags: set[str] = set()
    has_bullish_language = False
    has_bearish_language = False

    for article in articles:
        title = str(article.get("title") or "").casefold()
        theme = str(article.get("theme") or "uncertain").casefold()
        confidence = str(article.get("confidence") or "low").casefold()
        publication_status = str(
            article.get("publication_time_status") or "unknown"
        ).casefold()

        if publication_status != "known":
            flags.add("publication_time_unknown")
        if theme == "uncertain":
            flags.add("unclassified_headline")
        if confidence == "low":
            flags.add("low_rule_match_confidence")

        if any(term in title for term in PROMOTIONAL_TERMS):
            flags.add("promotional_or_guarantee_language")

        if any(term in title for term in BULLISH_TERMS):
            has_bullish_language = True
        if any(term in title for term in BEARISH_TERMS):
            has_bearish_language = True

    if has_bullish_language and has_bearish_language:
        flags.add("mixed_directional_headline_language")

    review_required = bool(flags)
    status = "review_required" if review_required else "no_flags_detected"

    return NewsSafetyAssessment(
        status=status,
        review_required=review_required,
        flags=tuple(sorted(flags)),
        checked_articles=len(articles),
        disclaimer=(
            "These automated checks identify limitations in headline context; "
            "they do not verify news, establish causation, predict prices, or "
            "provide buy/sell advice. No detected flags does not mean a trade "
            "is safe."
        ),
    )
