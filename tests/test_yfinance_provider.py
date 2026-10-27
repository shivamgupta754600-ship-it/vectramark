import pandas as pd
import pytest

from vectramark.core.exceptions import MarketDataError
from vectramark.infrastructure.market_data.yfinance_provider import (
    YahooFinanceProvider,
)


def sample_data() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=2, freq="D")

    return pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 101.0],
            "Close": [103.0, 104.0],
            "Volume": [1000, 1200],
        },
        index=index,
    )


def test_provider_calls_downloader_with_mapped_symbol() -> None:
    captured: dict[str, object] = {}

    def fake_downloader(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return sample_data()

    provider = YahooFinanceProvider(downloader=fake_downloader)
    result = provider.get_historical_data("XAUUSD", "1y", "1d")

    assert captured["tickers"] == "GC=F"
    assert captured["period"] == "1y"
    assert captured["interval"] == "1d"
    assert not result.empty


def test_provider_rejects_invalid_period() -> None:
    provider = YahooFinanceProvider(downloader=lambda **_: sample_data())

    with pytest.raises(MarketDataError):
        provider.get_historical_data("XAUUSD", "999y", "1d")


def test_provider_rejects_invalid_interval() -> None:
    provider = YahooFinanceProvider(downloader=lambda **_: sample_data())

    with pytest.raises(MarketDataError):
        provider.get_historical_data("XAUUSD", "1y", "999m")


def test_provider_rejects_empty_response() -> None:
    provider = YahooFinanceProvider(downloader=lambda **_: pd.DataFrame())

    with pytest.raises(MarketDataError, match="No historical data"):
        provider.get_historical_data("XAUUSD")


def test_provider_wraps_external_failure() -> None:
    def failing_downloader(**_: object) -> pd.DataFrame:
        raise TimeoutError("simulated timeout")

    provider = YahooFinanceProvider(downloader=failing_downloader)

    with pytest.raises(MarketDataError, match="provider failed"):
        provider.get_historical_data("XAUUSD")


def test_provider_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError):
        YahooFinanceProvider(timeout_seconds=0)
