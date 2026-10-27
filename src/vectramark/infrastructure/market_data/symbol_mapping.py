from dataclasses import dataclass

from vectramark.core.exceptions import MarketDataError


@dataclass(frozen=True, slots=True)
class ProviderInstrument:
    """Describe a market and its provider-specific instrument."""

    internal_symbol: str
    provider_symbol: str
    instrument_type: str
    description: str


_PROVIDER_INSTRUMENTS: dict[str, ProviderInstrument] = {
    "XAUUSD": ProviderInstrument(
        internal_symbol="XAUUSD",
        provider_symbol="GC=F",
        instrument_type="futures",
        description="Gold futures quoted in US dollars.",
    ),
    "XAGUSD": ProviderInstrument(
        internal_symbol="XAGUSD",
        provider_symbol="SI=F",
        instrument_type="futures",
        description="Silver futures quoted in US dollars.",
    ),
    "NASDAQ100": ProviderInstrument(
        internal_symbol="NASDAQ100",
        provider_symbol="^NDX",
        instrument_type="index",
        description="Nasdaq-100 index.",
    ),
    "SP500": ProviderInstrument(
        internal_symbol="SP500",
        provider_symbol="^GSPC",
        instrument_type="index",
        description="S&P 500 index.",
    ),
    "DJIA": ProviderInstrument(
        internal_symbol="DJIA",
        provider_symbol="^DJI",
        instrument_type="index",
        description="Dow Jones Industrial Average index.",
    ),
}


def get_provider_instrument(internal_symbol: str) -> ProviderInstrument:
    """Return the verified mapping configured for an internal market ID."""
    if not isinstance(internal_symbol, str):
        raise MarketDataError("Market symbol must be a string.")

    normalized_symbol = internal_symbol.strip().upper()

    if not normalized_symbol:
        raise MarketDataError("Market symbol cannot be empty.")

    instrument = _PROVIDER_INSTRUMENTS.get(normalized_symbol)

    if instrument is None:
        raise MarketDataError(f"Unsupported market symbol: {normalized_symbol}")

    return instrument


def list_provider_instruments() -> tuple[ProviderInstrument, ...]:
    """Return all configured provider instruments."""
    return tuple(_PROVIDER_INSTRUMENTS.values())
