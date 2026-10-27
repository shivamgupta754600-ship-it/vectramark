from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    """Metadata for a supported futures-market instrument."""

    symbol: str
    name: str
    category: str
    description: str


MARKETS: dict[str, Market] = {
    "XAUUSD": Market(
        symbol="XAUUSD",
        name="Gold Futures",
        category="precious_metal",
        description=(
            "Gold futures market data via Yahoo Finance ticker GC=F. "
            "This is futures data, not a spot gold quote."
        ),
    ),
    "XAGUSD": Market(
        symbol="XAGUSD",
        name="Silver Futures",
        category="precious_metal",
        description=(
            "Silver futures market data via Yahoo Finance ticker SI=F. "
            "This is futures data, not a spot silver quote."
        ),
    ),
    "NASDAQ100": Market(
        symbol="NASDAQ100",
        name="Nasdaq-100 Futures",
        category="equity_index",
        description=(
            "Nasdaq-100 futures market data via Yahoo Finance ticker NQ=F. "
            "This is futures data, not the cash Nasdaq-100 index."
        ),
    ),
    "SP500": Market(
        symbol="SP500",
        name="S&P 500 Futures",
        category="equity_index",
        description=(
            "S&P 500 futures market data via Yahoo Finance ticker ES=F. "
            "This is futures data, not the cash S&P 500 index."
        ),
    ),
    "DJIA": Market(
        symbol="DJIA",
        name="Dow Futures",
        category="equity_index",
        description=(
            "Dow futures market data via Yahoo Finance ticker YM=F. "
            "This is futures data, not the cash Dow Jones index."
        ),
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
