from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = ("Open", "High", "Low", "Close")

STALE_AFTER = {
    "1m": timedelta(minutes=10),
    "2m": timedelta(minutes=20),
    "5m": timedelta(minutes=45),
    "15m": timedelta(hours=2),
    "30m": timedelta(hours=4),
    "60m": timedelta(hours=8),
    "90m": timedelta(hours=12),
    "1h": timedelta(hours=8),
    "1d": timedelta(days=7),
    "5d": timedelta(days=14),
    "1wk": timedelta(days=21),
    "1mo": timedelta(days=62),
    "3mo": timedelta(days=125),
}


class MarketDataQualityAnalyzer:
    """Assess data integrity, history sufficiency, and timestamp freshness."""

    def assess(
        self,
        data: pd.DataFrame,
        interval: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        checked_at = now or datetime.now(UTC)

        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=UTC)
        else:
            checked_at = checked_at.astimezone(UTC)

        issues: list[str] = []
        warnings: list[str] = []

        candle_count = len(data) if isinstance(data, pd.DataFrame) else 0
        required_columns_present = isinstance(data, pd.DataFrame) and all(
            column in data.columns for column in REQUIRED_PRICE_COLUMNS
        )

        ohlc_valid = False

        if not isinstance(data, pd.DataFrame) or data.empty:
            issues.append("No market candles are available.")
        elif not required_columns_present:
            missing = [
                column
                for column in REQUIRED_PRICE_COLUMNS
                if column not in data.columns
            ]
            issues.append(f"Required OHLC columns are missing: {missing}.")
        else:
            prices = data.loc[:, REQUIRED_PRICE_COLUMNS].apply(
                pd.to_numeric,
                errors="coerce",
            )
            values = prices.to_numpy(dtype=float)

            finite_positive = bool(np.isfinite(values).all() and (values > 0).all())

            if finite_positive:
                high = prices["High"]
                low = prices["Low"]
                open_price = prices["Open"]
                close = prices["Close"]

                consistent_ohlc = bool(
                    (high >= low).all()
                    and (high >= open_price).all()
                    and (high >= close).all()
                    and (low <= open_price).all()
                    and (low <= close).all()
                )
            else:
                consistent_ohlc = False

            ohlc_valid = finite_positive and consistent_ohlc

            if not finite_positive:
                issues.append(
                    "OHLC data contains non-numeric, non-finite, "
                    "or non-positive prices."
                )
            elif not consistent_ohlc:
                issues.append("OHLC candle high/low relationships are inconsistent.")

        timestamps = pd.to_datetime(
            data.index if isinstance(data, pd.DataFrame) else [],
            utc=True,
            errors="coerce",
        )

        timestamps_valid = len(timestamps) > 0 and not timestamps.isna().any()
        duplicates_absent = timestamps_valid and not timestamps.duplicated().any()

        if not timestamps_valid:
            issues.append("Candle timestamps are missing or invalid.")
        elif not duplicates_absent:
            issues.append("Duplicate candle timestamps were detected.")

        last_candle_timestamp = None
        data_age_seconds = None
        fresh = False

        if timestamps_valid:
            latest_timestamp = timestamps.max().to_pydatetime()
            last_candle_timestamp = latest_timestamp
            data_age_seconds = (checked_at - latest_timestamp).total_seconds()

            if data_age_seconds < -300:
                issues.append(
                    "The latest candle timestamp is more than "
                    "five minutes in the future."
                )
            else:
                threshold = STALE_AFTER.get(interval, timedelta(days=7))
                fresh = (
                    data_age_seconds >= 0
                    and data_age_seconds <= threshold.total_seconds()
                )

                if not fresh:
                    warnings.append(
                        f"Latest candle exceeds the freshness threshold for "
                        f"interval '{interval}'."
                    )

        sufficient_history = candle_count >= 5

        if not sufficient_history:
            warnings.append(
                f"Only {candle_count} candle(s) are available; at least 5 "
                "are recommended for basic analysis."
            )

        structural_validity = ohlc_valid and timestamps_valid and duplicates_absent

        if not structural_validity:
            status = "invalid"
        elif warnings:
            status = "warning"
        else:
            status = "good"

        return {
            "checked_at": checked_at,
            "candle_count": candle_count,
            "last_candle_timestamp": last_candle_timestamp,
            "data_age_seconds": (
                max(0.0, data_age_seconds) if data_age_seconds is not None else None
            ),
            "status": status,
            "checks": {
                "ohlc_valid": ohlc_valid,
                "timestamps_valid": timestamps_valid,
                "duplicates_absent": duplicates_absent,
                "sufficient_history": sufficient_history,
                "freshness": fresh,
            },
            "issues": issues,
            "warnings": warnings,
        }
