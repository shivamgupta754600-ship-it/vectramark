import pytest

from vectramark.core.exceptions import MarketDataError
from vectramark.infrastructure.market_data.symbol_mapping import (
    get_provider_instrument,
    list_provider_instruments,
)


@pytest.mark.parametrize(
    ("symbol", "provider_symbol"),
    [
        ("XAUUSD", "GC=F"),
        ("XAGUSD", "SI=F"),
        ("NASDAQ100", "NQ=F"),
        ("SP500", "ES=F"),
        ("DJIA", "YM=F"),
    ],
)
def test_all_market_mappings_are_futures(symbol, provider_symbol):
    instrument = get_provider_instrument(symbol)

    assert instrument.internal_symbol == symbol
    assert instrument.provider_symbol == provider_symbol
    assert instrument.instrument_type == "futures"
    assert "futures" in instrument.description.lower()


def test_symbol_lookup_is_case_insensitive():
    assert get_provider_instrument("xauusd").internal_symbol == "XAUUSD"


@pytest.mark.parametrize("symbol", ["", "UNKNOWN", "../XAUUSD", None])
def test_unknown_symbol_is_rejected(symbol):
    with pytest.raises(MarketDataError):
        get_provider_instrument(symbol)


def test_all_five_instruments_are_configured():
    instruments = list_provider_instruments()

    assert len(instruments) == 5
    assert all(item.instrument_type == "futures" for item in instruments)


def test_cash_index_tickers_are_not_configured():
    configured_tickers = {
        instrument.provider_symbol for instrument in list_provider_instruments()
    }

    assert "^NDX" not in configured_tickers
    assert "^GSPC" not in configured_tickers
    assert "^DJI" not in configured_tickers
