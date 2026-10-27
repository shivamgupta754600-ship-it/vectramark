from datetime import UTC, datetime

import pandas as pd

from vectramark.domain.market_data.quality import MarketDataQualityAnalyzer


def valid_data() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=10,
        freq="D",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "Open": [100.0] * 10,
            "High": [102.0] * 10,
            "Low": [99.0] * 10,
            "Close": [101.0] * 10,
            "Volume": [1000.0] * 10,
        },
        index=timestamps,
    )


def test_recent_valid_data_has_good_quality() -> None:
    analyzer = MarketDataQualityAnalyzer()

    report = analyzer.assess(
        valid_data(),
        interval="1d",
        now=datetime(2026, 1, 12, tzinfo=UTC),
    )

    assert report["status"] == "good"
    assert report["candle_count"] == 10
    assert report["checks"]["ohlc_valid"] is True
    assert report["checks"]["timestamps_valid"] is True
    assert report["checks"]["duplicates_absent"] is True
    assert report["checks"]["sufficient_history"] is True
    assert report["checks"]["freshness"] is True
    assert report["issues"] == []
    assert report["warnings"] == []


def test_stale_data_produces_warning() -> None:
    analyzer = MarketDataQualityAnalyzer()

    report = analyzer.assess(
        valid_data(),
        interval="1d",
        now=datetime(2026, 2, 1, tzinfo=UTC),
    )

    assert report["status"] == "warning"
    assert report["checks"]["freshness"] is False
    assert any("freshness threshold" in warning for warning in report["warnings"])


def test_insufficient_history_produces_warning() -> None:
    analyzer = MarketDataQualityAnalyzer()

    report = analyzer.assess(
        valid_data().head(2),
        interval="1d",
        now=datetime(2026, 1, 2, 12, tzinfo=UTC),
    )

    assert report["status"] == "warning"
    assert report["checks"]["sufficient_history"] is False
    assert any("at least 5" in warning for warning in report["warnings"])


def test_inconsistent_ohlc_is_invalid() -> None:
    analyzer = MarketDataQualityAnalyzer()
    data = valid_data()
    data.loc[data.index[0], "High"] = 98.0

    report = analyzer.assess(
        data,
        interval="1d",
        now=datetime(2026, 1, 12, tzinfo=UTC),
    )

    assert report["status"] == "invalid"
    assert report["checks"]["ohlc_valid"] is False
    assert any("inconsistent" in issue for issue in report["issues"])


def test_duplicate_timestamps_are_invalid() -> None:
    analyzer = MarketDataQualityAnalyzer()
    data = valid_data()
    data.index = pd.DatetimeIndex(
        [
            *data.index[:-1],
            data.index[-2],
        ]
    )

    report = analyzer.assess(
        data,
        interval="1d",
        now=datetime(2026, 1, 12, tzinfo=UTC),
    )

    assert report["status"] == "invalid"
    assert report["checks"]["duplicates_absent"] is False
    assert any("Duplicate" in issue for issue in report["issues"])


def test_missing_ohlc_columns_are_invalid() -> None:
    analyzer = MarketDataQualityAnalyzer()
    data = valid_data().drop(columns=["High"])

    report = analyzer.assess(
        data,
        interval="1d",
        now=datetime(2026, 1, 12, tzinfo=UTC),
    )

    assert report["status"] == "invalid"
    assert report["checks"]["ohlc_valid"] is False
    assert any("missing" in issue for issue in report["issues"])
