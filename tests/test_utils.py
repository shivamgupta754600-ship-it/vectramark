import pandas as pd

from vectramark.core.utils import dataframe_is_empty, safe_float, utc_now


def test_safe_float_converts_valid_number():
    assert safe_float("123.45") == 123.45


def test_safe_float_rejects_invalid_value():
    assert safe_float("invalid") is None


def test_safe_float_rejects_infinity():
    assert safe_float(float("inf")) is None


def test_dataframe_is_empty():
    assert dataframe_is_empty(pd.DataFrame())
    assert dataframe_is_empty(None)
    assert not dataframe_is_empty(pd.DataFrame({"price": [100]}))


def test_utc_now_is_timezone_aware():
    timestamp = utc_now()

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset().total_seconds() == 0
