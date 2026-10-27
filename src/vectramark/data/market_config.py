from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    """Metadata for a supported market."""

    symbol: str
    name: str
    category: str
    description: str


MARKETS: dict[str, Market] = {
    "XAUUSD": Market(
        symbol="XAUUSD",
        name="Gold / US Dollar",
        category="precious_metal",
        description="Gold spot price quoted in US dollars.",
    ),
    "XAGUSD": Market(
        symbol="XAGUSD",
        name="Silver / US Dollar",
        category="precious_metal",
        description="Silver spot price quoted in US dollars.",
    ),
    "NASDAQ100": Market(
        symbol="NASDAQ100",
        name="Nasdaq-100",
        category="equity_index",
        description="Nasdaq-100 index; provider mapping must be verified.",
    ),
    "SP500": Market(
        symbol="SP500",
        name="S&P 500",
        category="equity_index",
        description="S&P 500 index; provider mapping must be verified.",
    ),
    "DJIA": Market(
        symbol="DJIA",
        name="Dow Jones Industrial Average",
        category="equity_index",
        description="Dow Jones Industrial Average; provider mapping must be verified.",
    ),
}


def get_market(symbol: str) -> Market:
    """Retrieve a supported market by its internal identifier."""
    key = symbol.strip().upper()

    try:
        return MARKETS[key]
    except KeyError as exc:
        supported = ", ".join(MARKETS)

        raise ValueError(
            f"Unsupported market '{symbol}'. Supported markets: {supported}"
        ) from exc


def list_markets() -> list[Market]:
    """Return all supported markets."""
    return list(MARKETS.values())
