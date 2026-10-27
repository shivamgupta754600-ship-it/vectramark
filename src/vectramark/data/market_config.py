from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    """Configuration describing a supported market."""

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
            "Gold futures quoted in US dollars. "
            "Configured Yahoo Finance ticker: GC=F. "
            "This instrument is not spot gold."
        ),
    ),
    "XAGUSD": Market(
        symbol="XAGUSD",
        name="Silver Futures",
        category="precious_metal",
        description=(
            "Silver futures quoted in US dollars. "
            "Configured Yahoo Finance ticker: SI=F. "
            "This instrument is not spot silver."
        ),
    ),
    "NASDAQ100": Market(
        symbol="NASDAQ100",
        name="Nasdaq-100 E-mini Futures",
        category="equity_index",
        description=(
            "E-mini Nasdaq-100 futures based on the Nasdaq-100 Index. "
            "Configured Yahoo Finance ticker: NQ=F. "
            "This is futures-market data, not the cash index."
        ),
    ),
    "SP500": Market(
        symbol="SP500",
        name="E-mini S&P 500 Futures",
        category="equity_index",
        description=(
            "E-mini S&P 500 index futures. "
            "Configured Yahoo Finance ticker: ES=F. "
            "This is futures-market data, not the cash index."
        ),
    ),
    "DJIA": Market(
        symbol="DJIA",
        name="E-mini Dow Futures",
        category="equity_index",
        description=(
            "Dow Jones index futures. "
            "Configured Yahoo Finance ticker: YM=F. "
            "This is futures-market data, not the cash index."
        ),
    ),
}


def get_market(symbol: str) -> Market:
    """Return configuration for a supported internal market symbol."""
    if not isinstance(symbol, str):
        raise ValueError("Market symbol must be a string.")

    normalized_symbol = symbol.strip().upper()

    if not normalized_symbol:
        raise ValueError("Market symbol cannot be empty.")

    try:
        return MARKETS[normalized_symbol]
    except KeyError as exc:
        raise ValueError("Unsupported market symbol.") from exc


def list_markets() -> list[Market]:
    """Return all configured markets."""
    return list(MARKETS.values())
