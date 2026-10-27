import pytest
from pydantic import ValidationError

from vectramark.api.schemas.market import MarketRequest


def test_valid_market_request():
    request = MarketRequest(symbol="XAUUSD")

    assert request.symbol == "XAUUSD"


def test_invalid_market_is_rejected():
    with pytest.raises(ValidationError):
        MarketRequest(symbol="UNKNOWN")


def test_extra_request_fields_are_rejected():
    with pytest.raises(ValidationError):
        MarketRequest(
            symbol="XAUUSD",
            unexpected_field="invalid",
        )
