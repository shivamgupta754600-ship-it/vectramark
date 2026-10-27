import pytest

from vectramark.api.routes.market_data import build_instrument_metadata
from vectramark.api.schemas.market_data import HistoricalMarketDataResponse
from vectramark.infrastructure.market_data.symbol_mapping import (
    DATA_PROVIDER_NAME,
)


@pytest.mark.parametrize(
    ("symbol", "provider_symbol", "expected_name"),
    [
        ("XAUUSD", "GC=F", "Gold Futures"),
        ("XAGUSD", "SI=F", "Silver Futures"),
        ("NASDAQ100", "NQ=F", "Nasdaq-100 E-mini Futures"),
        ("SP500", "ES=F", "E-mini S&P 500 Futures"),
        ("DJIA", "YM=F", "E-mini Dow Futures"),
    ],
)
def test_instrument_metadata_matches_configured_futures(
    symbol: str,
    provider_symbol: str,
    expected_name: str,
) -> None:
    metadata = build_instrument_metadata(symbol)

    assert metadata.internal_symbol == symbol
    assert metadata.provider_symbol == provider_symbol
    assert metadata.instrument_type == "futures"
    assert metadata.provider_name == DATA_PROVIDER_NAME
    assert metadata.name == expected_name
    assert "futures" in metadata.description.lower()


def test_nasdaq_metadata_distinguishes_futures_from_cash_index() -> None:
    metadata = build_instrument_metadata("NASDAQ100")

    assert metadata.name == "Nasdaq-100 E-mini Futures"
    assert metadata.provider_symbol == "NQ=F"
    assert metadata.instrument_type == "futures"
    assert "Nasdaq-100 Index" in metadata.description
    assert "not the cash index" in metadata.description.lower()


def test_historical_response_can_serialize_instrument_metadata() -> None:
    metadata = build_instrument_metadata("NASDAQ100")

    response = HistoricalMarketDataResponse(
        symbol="NASDAQ100",
        period="5d",
        interval="1d",
        count=0,
        data=[],
        instrument=metadata,
    )

    payload = response.model_dump()

    assert payload["instrument"]["internal_symbol"] == "NASDAQ100"
    assert payload["instrument"]["name"] == "Nasdaq-100 E-mini Futures"
    assert payload["instrument"]["provider_symbol"] == "NQ=F"
    assert payload["instrument"]["instrument_type"] == "futures"
