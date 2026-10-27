"""Technical indicators for historical market price analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_close(close: pd.Series) -> pd.Series:
    """Validate and return a numeric closing-price series."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")

    if close.empty:
        raise ValueError("close cannot be empty.")

    if close.isna().any():
        raise ValueError("close cannot contain missing values.")

    values = pd.to_numeric(close, errors="raise").astype(float)

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("close must contain only finite values.")

    if (values <= 0).any():
        raise ValueError("Closing prices must be greater than zero.")

    return values


def _validate_window(window: int, name: str = "window") -> None:
    """Validate a positive integer window."""
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError(f"{name} must be an integer.")

    if window < 1:
        raise ValueError(f"{name} must be greater than zero.")


def calculate_sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate the Simple Moving Average."""
    _validate_window(window)
    prices = _validate_close(close)

    return (
        prices.rolling(window=window, min_periods=window).mean().rename(f"SMA_{window}")
    )


def calculate_ema(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate the Exponential Moving Average."""
    _validate_window(window)
    prices = _validate_close(close)

    return (
        prices.ewm(
            span=window,
            adjust=False,
            min_periods=window,
        )
        .mean()
        .rename(f"EMA_{window}")
    )


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing.

    RSI ranges from 0 to 100. The first window observations are
    unavailable because the calculation needs price changes.
    """
    _validate_window(window)
    prices = _validate_close(close)

    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    alpha = 1 / window

    average_gain = gains.ewm(
        alpha=alpha,
        adjust=False,
        min_periods=window,
    ).mean()

    average_loss = losses.ewm(
        alpha=alpha,
        adjust=False,
        min_periods=window,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    # No losses but some gains: maximum RSI.
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)

    # No gains and no losses: neutral RSI for a flat series.
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)

    return rsi.clip(lower=0, upper=100).rename(f"RSI_{window}")


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calculate MACD, signal line, and histogram."""
    _validate_window(fast, "fast")
    _validate_window(slow, "slow")
    _validate_window(signal, "signal")

    if fast >= slow:
        raise ValueError("fast must be smaller than slow.")

    prices = _validate_close(close)

    fast_ema = prices.ewm(
        span=fast,
        adjust=False,
        min_periods=fast,
    ).mean()

    slow_ema = prices.ewm(
        span=slow,
        adjust=False,
        min_periods=slow,
    ).mean()

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False,
        min_periods=signal,
    ).mean()

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_signal": signal_line,
            "MACD_histogram": macd_line - signal_line,
        },
        index=prices.index,
    )
