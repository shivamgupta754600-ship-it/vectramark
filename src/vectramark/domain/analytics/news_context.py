"""Transparent, rule-based classification of market news headlines."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NewsContext:
    """Explainable classification of one news headline."""

    theme: str
    confidence: str
    evidence: tuple[str, ...]
    market_relevance: str
    interpretation: str


# Ordered rules make classification deterministic when multiple themes match.
THEME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "interest_rates",
        (
            "interest rate",
            "interest rates",
            "rate cut",
            "rate hike",
            "fed funds",
            "federal reserve",
            "central bank",
            "monetary policy",
            "hawkish",
            "dovish",
        ),
    ),
    (
        "inflation",
        (
            "inflation",
            "consumer price index",
            "cpi",
            "producer price index",
            "ppi",
            "price pressures",
        ),
    ),
    (
        "employment",
        (
            "employment",
            "unemployment",
            "jobs report",
            "payrolls",
            "jobless claims",
            "labor market",
        ),
    ),
    (
        "economic_growth",
        (
            "economic growth",
            "recession",
            "gdp",
            "economic slowdown",
            "manufacturing data",
            "consumer spending",
        ),
    ),
    (
        "currency_and_yields",
        (
            "us dollar",
            "u.s. dollar",
            "dollar index",
            "treasury yields",
            "bond yields",
            "yield rises",
            "yield falls",
        ),
    ),
    (
        "corporate_earnings",
        (
            "earnings",
            "quarterly results",
            "profit forecast",
            "revenue forecast",
            "earnings outlook",
        ),
    ),
    (
        "industrial_demand",
        (
            "industrial demand",
            "solar demand",
            "silver demand",
            "industrial production",
            "manufacturing demand",
        ),
    ),
    (
        "geopolitics",
        (
            "geopolitical",
            "geopolitics",
            "military conflict",
            "trade war",
            "sanctions",
            "ceasefire",
        ),
    ),
)

MARKET_RELEVANCE: dict[str, dict[str, str]] = {
    "XAUUSD": {
        "interest_rates": (
            "Interest-rate expectations can affect gold's opportunity cost."
        ),
        "inflation": (
            "Inflation data can influence rate expectations and demand for gold."
        ),
        "currency_and_yields": (
            "US dollar and Treasury-yield changes can affect gold pricing."
        ),
        "geopolitics": (
            "Geopolitical developments may affect demand for perceived safe havens."
        ),
    },
    "XAGUSD": {
        "interest_rates": (
            "Rate expectations can influence the US dollar and precious metals."
        ),
        "inflation": (
            "Inflation data may affect monetary-policy expectations and metals."
        ),
        "currency_and_yields": (
            "US dollar and yield changes can influence silver pricing."
        ),
        "industrial_demand": (
            "Industrial demand is relevant because silver has industrial uses."
        ),
        "economic_growth": (
            "Growth expectations may affect industrial demand for silver."
        ),
        "geopolitics": (
            "Geopolitical events may affect risk sentiment and metals markets."
        ),
    },
    "NASDAQ100": {
        "interest_rates": (
            "Rate expectations can affect equity valuations, especially growth stocks."
        ),
        "inflation": (
            "Inflation data can change expectations for interest rates and valuations."
        ),
        "corporate_earnings": (
            "Earnings reports can influence expectations for index constituents."
        ),
        "economic_growth": (
            "Growth data can affect expectations for corporate revenue and profits."
        ),
    },
    "SP500": {
        "interest_rates": ("Rate expectations can influence broad equity valuations."),
        "inflation": (
            "Inflation data can influence monetary policy and company costs."
        ),
        "corporate_earnings": (
            "Earnings reports can affect expectations for index constituents."
        ),
        "economic_growth": (
            "Growth data can influence expectations for company revenues and profits."
        ),
    },
    "DJIA": {
        "interest_rates": (
            "Rate expectations can affect equity valuations and borrowing costs."
        ),
        "inflation": (
            "Inflation data can influence monetary policy and company costs."
        ),
        "corporate_earnings": (
            "Earnings reports can affect expectations for index constituents."
        ),
        "economic_growth": (
            "Growth data can affect expectations for companies "
            "represented in the index."
        ),
    },
}


def analyze_headline(
    symbol: str,
    title: str,
    summary: str | None = None,
) -> NewsContext:
    """Classify a headline using matched phrases, without predicting prices."""

    normalized_symbol = symbol.strip().upper()
    text = f"{title} {summary or ''}".casefold()
    matches: list[tuple[str, tuple[str, ...]]] = []

    for theme, phrases in THEME_RULES:
        evidence = tuple(phrase for phrase in phrases if phrase in text)
        if evidence:
            matches.append((theme, evidence))

    if not matches:
        return NewsContext(
            theme="uncertain",
            confidence="low",
            evidence=(),
            market_relevance=(
                "No configured market-news theme could be identified reliably."
            ),
            interpretation=(
                "The headline needs manual review. Its market impact cannot "
                "be determined from the available text."
            ),
        )

    theme, evidence = max(matches, key=lambda item: len(item[1]))

    # Confidence describes rule-match strength, not factual certainty.
    confidence = (
        "high" if len(evidence) >= 3 else ("medium" if len(evidence) >= 2 else "low")
    )

    relevance = MARKET_RELEVANCE.get(normalized_symbol, {}).get(theme)
    if relevance is None:
        relevance = (
            f"The headline matches the {theme.replace('_', ' ')} theme, "
            "but its relevance to this market is not configured."
        )

    return NewsContext(
        theme=theme,
        confidence=confidence,
        evidence=evidence,
        market_relevance=relevance,
        interpretation=(
            f"The text matches the {theme.replace('_', ' ')} theme. "
            "This is a rule-based interpretation of the headline, not "
            "confirmation of the reported event or a prediction of price direction."
        ),
    )
