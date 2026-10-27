import pandas as pd
import pytest

from vectramark.application.market_data_service import MarketDataService
from vectramark.core.exceptions import MarketDataError
from vectramark.infrastructure.storage.market_store import LocalMarketStore


class FakeProvider:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.calls = 0

    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.calls += 1
        return self.data.copy()


def valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 101.0],
            "Close": [103.0, 104.0],
            "Volume": [1000, 1200],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="D"),
    )


def test_service_fetches_validates_and_saves(tmp_path) -> None:
    provider = FakeProvider(valid_data())
    store = LocalMarketStore(data_dir=tmp_path)
    service = MarketDataService(provider=provider, store=store)

    result = service.fetch_and_store("XAUUSD")

    assert provider.calls == 1
    assert len(result) == 2
    assert (tmp_path / "XAUUSD.csv").exists()


def test_service_can_load_saved_data(tmp_path) -> None:
    provider = FakeProvider(valid_data())
    store = LocalMarketStore(data_dir=tmp_path)
    service = MarketDataService(provider=provider, store=store)

    service.fetch_and_store("XAUUSD")
    result = service.load_saved_data("XAUUSD")

    assert len(result) == 2


def test_service_rejects_unknown_market(tmp_path) -> None:
    provider = FakeProvider(valid_data())
    store = LocalMarketStore(data_dir=tmp_path)
    service = MarketDataService(provider=provider, store=store)

    with pytest.raises((MarketDataError, ValueError, KeyError)):
        service.fetch_and_store("UNKNOWN")

    assert provider.calls == 0


def test_service_does_not_save_invalid_data(tmp_path) -> None:
    data = valid_data()
    data.iloc[0, data.columns.get_loc("Close")] = -1.0

    provider = FakeProvider(data)
    store = LocalMarketStore(data_dir=tmp_path)
    service = MarketDataService(provider=provider, store=store)

    with pytest.raises(Exception):
        service.fetch_and_store("XAUUSD")

    assert not (tmp_path / "XAUUSD.csv").exists()
