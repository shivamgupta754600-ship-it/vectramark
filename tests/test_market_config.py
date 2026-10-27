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


def test_market_lookup_is_case_insensitive():
    assert get_market("xauusd") == get_market("XAUUSD")


def test_unsupported_market_raises_error():
    with pytest.raises(ValueError, match="Unsupported market"):
        get_market("UNKNOWN")


def test_list_markets_returns_all_markets():
    assert len(list_markets()) == 5
