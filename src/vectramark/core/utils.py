import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def dataframe_is_empty(data: pd.DataFrame | None) -> bool:
    """Check whether a DataFrame is missing or empty."""
    return data is None or data.empty


def safe_float(value: Any) -> float | None:
    """Convert a value to a finite float, or return None."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return result if math.isfinite(result) else None
