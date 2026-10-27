from dataclasses import dataclass

from vectramark.core.exceptions import MarketDataError

DATA_PROVIDER_NAME = "Yahoo Finance via yfinance"


@dataclass(frozen=True)
class ProviderInstrument:
    """Provider-specific identity for an internal market."""

    internal_symbol: str
    provider_symbol: str
    instrument_type: str
    description: str


PROVIDER_INSTRUMENTS: dict[str, ProviderInstrument] = {
    "XAUUSD": ProviderInstrument(
        internal_symbol="XAUUSD",
        provider_symbol="GC=F",
        instrument_type="futures",
        description=(
            "Gold futures from Yahoo Finance using GC=F. This is not spot gold."
        ),
    ),
    "XAGUSD": ProviderInstrument(
        internal_symbol="XAGUSD",
        provider_symbol="SI=F",
        instrument_type="futures",
        description=(
            "Silver futures from Yahoo Finance using SI=F. This is not spot silver."
        ),
    ),
    "NASDAQ100": ProviderInstrument(
        internal_symbol="NASDAQ100",
        provider_symbol="NQ=F",
        instrument_type="futures",
        description=(
            "E-mini Nasdaq-100 futures based on the Nasdaq-100 Index, "
            "using Yahoo Finance ticker NQ=F. "
            "This is not the cash index."
        ),
    ),
    "SP500": ProviderInstrument(
        internal_symbol="SP500",
        provider_symbol="ES=F",
        instrument_type="futures",
        description=(
            "E-mini S&P 500 futures using Yahoo Finance ticker ES=F. "
            "This is not the cash index."
        ),
    ),
    "DJIA": ProviderInstrument(
        internal_symbol="DJIA",
        provider_symbol="YM=F",
        instrument_type="futures",
        description=(
            "Dow Jones futures using Yahoo Finance ticker YM=F. "
            "This is not the cash index."
        ),
    ),
}


def get_provider_instrument(internal_symbol: str) -> ProviderInstrument:
    """Resolve an internal market symbol to its configured provider ticker."""
    if not isinstance(internal_symbol, str):
        raise MarketDataError("Market symbol must be a string.")

    normalized_symbol = internal_symbol.strip().upper()

    if not normalized_symbol:
        raise MarketDataError("Market symbol cannot be empty.")

    try:
        return PROVIDER_INSTRUMENTS[normalized_symbol]
    except KeyError as exc:
        raise MarketDataError("Unsupported provider instrument.") from exc


def list_provider_instruments() -> list[ProviderInstrument]:
    """Return all configured provider instruments."""
    return list(PROVIDER_INSTRUMENTS.values())
