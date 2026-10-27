import pytest

from vectramark.data.market_config import MARKETS, get_market, list_markets


def test_five_markets_are_configured():
    assert len(MARKETS) == 5


@pytest.mark.parametrize(
    "symbol",
    ["XAUUSD", "XAGUSD", "NASDAQ100", "SP500", "DJIA"],
)
def test_supported_markets(symbol):
    market = get_market(symbol)

    assert market.symbol == symbol
    assert market.name
    assert "futures" in market.name.lower()
    assert "futures" in market.description.lower()


def test_market_lookup_is_case_insensitive():
    assert get_market("xauusd") == get_market("XAUUSD")


def test_unsupported_market_raises_error():
    with pytest.raises(ValueError, match="Unsupported market"):
        get_market("UNKNOWN")


def test_list_markets_returns_all_markets():
    assert len(list_markets()) == 5


@pytest.mark.parametrize(
    ("symbol", "provider_ticker"),
    [
        ("XAUUSD", "GC=F"),
        ("XAGUSD", "SI=F"),
        ("NASDAQ100", "NQ=F"),
        ("SP500", "ES=F"),
        ("DJIA", "YM=F"),
    ],
)
def test_market_description_identifies_futures_ticker(symbol, provider_ticker):
    market = get_market(symbol)

    assert provider_ticker in market.description
