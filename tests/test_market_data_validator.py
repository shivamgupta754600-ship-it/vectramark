import pandas as pd
import pytest

from vectramark.core.exceptions import DataValidationError
from vectramark.domain.market_data.validator import MarketDataValidator


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


def test_valid_data_passes_and_uses_utc_timestamps() -> None:
    result = MarketDataValidator().validate(valid_data())

    assert len(result) == 2
    assert str(result.index.tz) == "UTC"


def test_input_is_not_modified() -> None:
    original = valid_data()
    before = original.copy(deep=True)

    MarketDataValidator().validate(original)

    pd.testing.assert_frame_equal(original, before)


def test_missing_price_column_is_rejected() -> None:
    data = valid_data().drop(columns=["Close"])

    with pytest.raises(DataValidationError, match="Missing required"):
        MarketDataValidator().validate(data)


def test_negative_price_is_rejected() -> None:
    data = valid_data()
    data.iloc[0, data.columns.get_loc("Close")] = -10.0

    with pytest.raises(DataValidationError, match="positive prices"):
        MarketDataValidator().validate(data)


def test_inconsistent_high_is_rejected() -> None:
    data = valid_data()
    data.iloc[0, data.columns.get_loc("High")] = 98.0

    with pytest.raises(DataValidationError, match="High cannot"):
        MarketDataValidator().validate(data)


def test_duplicate_timestamps_are_rejected() -> None:
    data = valid_data()
    data.index = pd.DatetimeIndex(["2026-01-01", "2026-01-01"])

    with pytest.raises(DataValidationError, match="duplicate timestamps"):
        MarketDataValidator().validate(data)


def test_infinite_price_is_rejected() -> None:
    data = valid_data()
    data.iloc[0, data.columns.get_loc("Close")] = float("inf")

    with pytest.raises(DataValidationError, match="non-finite"):
        MarketDataValidator().validate(data)


def test_missing_volume_is_allowed() -> None:
    data = valid_data().drop(columns=["Volume"])

    result = MarketDataValidator().validate(data)

    assert "Volume" not in result.columns


def test_negative_volume_is_rejected() -> None:
    data = valid_data()
    data.iloc[0, data.columns.get_loc("Volume")] = -1

    with pytest.raises(DataValidationError, match="negative"):
        MarketDataValidator().validate(data)
