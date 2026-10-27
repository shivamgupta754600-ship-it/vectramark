from dataclasses import dataclass

from vectramark.core.exceptions import MarketDataError


@dataclass(frozen=True, slots=True)
class ProviderInstrument:
    """Describe a futures-market instrument at its data provider."""

    internal_symbol: str
    provider_symbol: str
    instrument_type: str
    description: str


_PROVIDER_INSTRUMENTS: dict[str, ProviderInstrument] = {
    "XAUUSD": ProviderInstrument(
        internal_symbol="XAUUSD",
        provider_symbol="GC=F",
        instrument_type="futures",
        description="Gold futures; Yahoo Finance ticker GC=F.",
    ),
    "XAGUSD": ProviderInstrument(
        internal_symbol="XAGUSD",
        provider_symbol="SI=F",
        instrument_type="futures",
        description="Silver futures; Yahoo Finance ticker SI=F.",
    ),
    "NASDAQ100": ProviderInstrument(
        internal_symbol="NASDAQ100",
        provider_symbol="NQ=F",
        instrument_type="futures",
        description="E-mini Nasdaq-100 futures; Yahoo Finance ticker NQ=F.",
    ),
    "SP500": ProviderInstrument(
        internal_symbol="SP500",
        provider_symbol="ES=F",
        instrument_type="futures",
        description="E-mini S&P 500 futures; Yahoo Finance ticker ES=F.",
    ),
    "DJIA": ProviderInstrument(
        internal_symbol="DJIA",
        provider_symbol="YM=F",
        instrument_type="futures",
        description="E-mini Dow futures; Yahoo Finance ticker YM=F.",
    ),
}


def get_provider_instrument(internal_symbol: str) -> ProviderInstrument:
    """Return the configured futures instrument for an internal market ID."""
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
    """Return all configured futures instruments."""
    return tuple(_PROVIDER_INSTRUMENTS.values())
