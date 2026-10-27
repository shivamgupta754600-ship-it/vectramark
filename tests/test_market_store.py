import pandas as pd
import pytest

from vectramark.core.exceptions import (
    DataValidationError,
    MarketDataError,
)
from vectramark.infrastructure.storage.market_store import LocalMarketStore


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


def test_save_and_load_round_trip(tmp_path) -> None:
    store = LocalMarketStore(data_dir=tmp_path)

    saved_path = store.save("XAUUSD", valid_data())
    loaded = store.load("XAUUSD")

    assert saved_path.exists()
    assert saved_path.parent == tmp_path.resolve()
    assert len(loaded) == 2
    assert list(loaded.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]


def test_unsupported_symbol_is_rejected(tmp_path) -> None:
    store = LocalMarketStore(data_dir=tmp_path)

    with pytest.raises(MarketDataError, match="Unsupported"):
        store.save("../unsafe", valid_data())


def test_invalid_data_is_not_saved(tmp_path) -> None:
    store = LocalMarketStore(data_dir=tmp_path)
    data = valid_data()
    data.iloc[0, data.columns.get_loc("Close")] = 0.0

    with pytest.raises(DataValidationError):
        store.save("XAUUSD", data)

    assert not (tmp_path / "XAUUSD.csv").exists()


def test_missing_saved_data_is_reported(tmp_path) -> None:
    store = LocalMarketStore(data_dir=tmp_path)

    with pytest.raises(MarketDataError, match="No saved market data"):
        store.load("XAUUSD")
