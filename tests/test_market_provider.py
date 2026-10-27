import inspect

import pandas as pd

from vectramark.infrastructure.market_data.provider import MarketDataProvider


def test_provider_is_abstract():
    assert inspect.isabstract(MarketDataProvider)


def test_provider_declares_historical_data_method():
    assert hasattr(MarketDataProvider, "get_historical_data")

    signature = inspect.signature(MarketDataProvider.get_historical_data)

    assert "symbol" in signature.parameters
    assert "period" in signature.parameters
    assert "interval" in signature.parameters


def test_provider_contract_uses_dataframe_return_annotation():
    annotation = MarketDataProvider.get_historical_data.__annotations__["return"]

    assert annotation == pd.DataFrame
