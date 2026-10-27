import numpy as np
import pandas as pd

from vectramark.core.exceptions import DataValidationError

REQUIRED_PRICE_COLUMNS = ("Open", "High", "Low", "Close")
OPTIONAL_COLUMNS = ("Volume",)


class MarketDataValidator:
    """Validate the structure and integrity of OHLCV market data."""

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a normalized copy of valid market data."""
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError("Market data must be a pandas DataFrame.")

        if data.empty:
            raise DataValidationError("Market data cannot be empty.")

        if not data.columns.is_unique:
            raise DataValidationError("Market data contains duplicate column names.")

        result = data.copy()

        # Normalize column labels without silently overwriting collisions.
        if not all(isinstance(column, str) for column in result.columns):
            raise DataValidationError("All market data column names must be strings.")

        result.columns = [column.strip().title() for column in result.columns]

        if not result.columns.is_unique:
            raise DataValidationError(
                "Column normalization created duplicate column names."
            )

        missing_columns = [
            column for column in REQUIRED_PRICE_COLUMNS if column not in result.columns
        ]

        if missing_columns:
            raise DataValidationError(
                f"Missing required price columns: {missing_columns}"
            )

        # A numeric index can represent nanoseconds accidentally. Reject it
        # rather than interpreting arbitrary integers as market timestamps.
        if pd.api.types.is_numeric_dtype(result.index.dtype):
            raise DataValidationError(
                "Market data index must contain timestamps, not numeric values."
            )

        try:
            timestamps = pd.to_datetime(
                result.index,
                errors="coerce",
                utc=True,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError(
                "Market data timestamps could not be parsed."
            ) from exc

        if timestamps.isna().any():
            raise DataValidationError("Market data contains invalid timestamps.")

        result.index = pd.DatetimeIndex(timestamps, name="timestamp")

        if result.index.has_duplicates:
            raise DataValidationError("Market data contains duplicate timestamps.")

        result = result.sort_index()

        for column in REQUIRED_PRICE_COLUMNS:
            try:
                result[column] = pd.to_numeric(
                    result[column],
                    errors="raise",
                )
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"{column} contains non-numeric values."
                ) from exc

            values = result[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise DataValidationError(
                    f"{column} contains missing or non-finite values."
                )

            if (values <= 0).any():
                raise DataValidationError(f"{column} must contain positive prices.")

        # OHLC consistency checks.
        if (result["High"] < result["Low"]).any():
            raise DataValidationError("High cannot be lower than Low.")

        if (result["High"] < result["Open"]).any():
            raise DataValidationError("High cannot be lower than Open.")

        if (result["High"] < result["Close"]).any():
            raise DataValidationError("High cannot be lower than Close.")

        if (result["Low"] > result["Open"]).any():
            raise DataValidationError("Low cannot be higher than Open.")

        if (result["Low"] > result["Close"]).any():
            raise DataValidationError("Low cannot be higher than Close.")

        if "Volume" in result.columns:
            try:
                result["Volume"] = pd.to_numeric(
                    result["Volume"],
                    errors="raise",
                )
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    "Volume contains non-numeric values."
                ) from exc

            volume = result["Volume"].to_numpy(dtype=float)
            finite_volume = np.isfinite(volume)

            if np.isinf(volume).any():
                raise DataValidationError("Volume contains infinite values.")

            if (volume[finite_volume] < 0).any():
                raise DataValidationError("Volume cannot be negative.")

        # Volume may legitimately be unavailable for some indices.
        # Do not fabricate volume values when the provider omits them.
        for column in OPTIONAL_COLUMNS:
            if column not in result.columns:
                continue

        return result
