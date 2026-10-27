import pytest

from vectramark.core.exceptions import MarketDataError
from vectramark.infrastructure.market_data.symbol_mapping import (
    get_provider_instrument,
    list_provider_instruments,
)


def test_gold_mapping_is_futures() -> None:
    instrument = get_provider_instrument("XAUUSD")

    assert instrument.provider_symbol == "GC=F"
    assert instrument.instrument_type == "futures"


def test_silver_mapping_is_futures() -> None:
    instrument = get_provider_instrument("XAGUSD")

    assert instrument.provider_symbol == "SI=F"
    assert instrument.instrument_type == "futures"


@pytest.mark.parametrize(
    ("symbol", "provider_symbol"),
    [
        ("NASDAQ100", "^NDX"),
        ("SP500", "^GSPC"),
        ("DJIA", "^DJI"),
    ],
)
def test_index_mappings(
    symbol: str,
    provider_symbol: str,
) -> None:
    instrument = get_provider_instrument(symbol)

    assert instrument.provider_symbol == provider_symbol
    assert instrument.instrument_type == "index"


def test_symbol_lookup_is_case_insensitive() -> None:
    assert get_provider_instrument("xauusd").internal_symbol == "XAUUSD"


@pytest.mark.parametrize("symbol", ["", "UNKNOWN", "../XAUUSD", None])
def test_unknown_symbol_is_rejected(symbol: str | None) -> None:
    with pytest.raises(MarketDataError):
        get_provider_instrument(symbol)  # type: ignore[arg-type]


def test_all_five_instruments_are_configured() -> None:
    assert len(list_provider_instruments()) == 5
