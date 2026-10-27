"""Rule-based, explainable market-bias calculations."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from vectramark.domain.analytics.technical_indicators import (
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)

BiasDirection = Literal["bullish", "bearish", "neutral"]
SignalDirection = Literal["bullish", "bearish", "neutral", "unavailable"]
BiasTimeframe = Literal["daily", "weekly", "monthly", "yearly"]

TIMEFRAME_LOOKBACKS: dict[str, int] = {
    "daily": 1,
    "weekly": 5,
    "monthly": 21,
    "yearly": 252,
}

RETURN_THRESHOLDS_PERCENT: dict[str, float] = {
    "daily": 0.1,
    "weekly": 0.3,
    "monthly": 1.0,
    "yearly": 5.0,
}

MIN_DIRECTIONAL_SIGNALS = 3


@dataclass(frozen=True)
class BiasSignal:
    """One transparent indicator contributing to market-bias analysis."""

    name: str
    value: float | None
    unit: str
    direction: SignalDirection
    explanation: str


@dataclass(frozen=True)
class MarketBiasResult:
    """Complete market-bias analysis result."""

    timeframe: BiasTimeframe
    as_of: pd.Timestamp
    latest_close: float
    observations: int
    bias: BiasDirection
    score: int
    explanation: str
    signals: tuple[BiasSignal, ...]


def _value_or_none(series: pd.Series) -> float | None:
    """Return the latest finite indicator value, if one exists."""

    if series.empty:
        return None

    value = series.iloc[-1]

    if pd.isna(value):
        return None

    number = float(value)

    if not np.isfinite(number):
        return None

    return number


def _unavailable_signal(
    name: str,
    unit: str,
    explanation: str,
) -> BiasSignal:
    """Represent an indicator that cannot yet be calculated."""

    return BiasSignal(
        name=name,
        value=None,
        unit=unit,
        direction="unavailable",
        explanation=explanation,
    )


def _trend_signal(
    name: str,
    close: float,
    average: float | None,
    average_name: str,
) -> BiasSignal:
    """Compare the latest close with a moving average."""

    if average is None:
        return _unavailable_signal(
            name=name,
            unit="percent",
            explanation=f"Insufficient history to calculate {average_name}.",
        )

    difference_percent = ((close / average) - 1.0) * 100.0

    if difference_percent > 0.05:
        direction: SignalDirection = "bullish"
        explanation = f"Close is {difference_percent:.2f}% above {average_name}."
    elif difference_percent < -0.05:
        direction = "bearish"
        explanation = f"Close is {abs(difference_percent):.2f}% below {average_name}."
    else:
        direction = "neutral"
        explanation = f"Close is within 0.05% of {average_name}."

    return BiasSignal(
        name=name,
        value=difference_percent,
        unit="percent",
        direction=direction,
        explanation=explanation,
    )


def calculate_market_bias(
    closing_prices: pd.Series,
    timeframe: BiasTimeframe = "daily",
) -> MarketBiasResult:
    """Estimate market bias from historical daily closing prices.

    This is a transparent technical-indicator heuristic, not a trained
    predictive model or a guarantee of future price direction.

    The horizon-return lookback uses trading observations:
    daily=1, weekly=5, monthly=21, yearly=252.
    """

    if timeframe not in TIMEFRAME_LOOKBACKS:
        raise ValueError("timeframe must be daily, weekly, monthly, or yearly.")

    if not isinstance(closing_prices, pd.Series):
        raise TypeError("closing_prices must be a pandas Series.")

    if closing_prices.empty:
        raise ValueError("closing_prices cannot be empty.")

    if closing_prices.isna().any():
        raise ValueError("closing_prices cannot contain missing values.")

    prices = pd.to_numeric(
        closing_prices.copy(),
        errors="raise",
    ).astype(float)

    values = prices.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("closing_prices must contain finite values.")

    if (values <= 0).any():
        raise ValueError("Closing prices must be greater than zero.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("closing_prices index must be a DatetimeIndex.")

    if prices.index.hasnans:
        raise ValueError("closing_prices index cannot contain missing dates.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("closing_prices must be ordered by date.")

    if prices.index.has_duplicates:
        raise ValueError("closing_prices index cannot contain duplicate dates.")

    latest_close = float(prices.iloc[-1])

    sma = calculate_sma(prices, window=20)
    ema = calculate_ema(prices, window=20)
    rsi = calculate_rsi(prices, window=14)
    macd = calculate_macd(prices)

    signals: list[BiasSignal] = [
        _trend_signal(
            name="price_vs_sma20",
            close=latest_close,
            average=_value_or_none(sma),
            average_name="the 20-period SMA",
        ),
        _trend_signal(
            name="price_vs_ema20",
            close=latest_close,
            average=_value_or_none(ema),
            average_name="the 20-period EMA",
        ),
    ]

    rsi_value = _value_or_none(rsi)

    if rsi_value is None:
        signals.append(
            _unavailable_signal(
                name="rsi14",
                unit="index",
                explanation="Insufficient history to calculate RSI(14).",
            )
        )
    elif rsi_value >= 70:
        signals.append(
            BiasSignal(
                name="rsi14",
                value=rsi_value,
                unit="index",
                direction="neutral",
                explanation=(
                    f"RSI is {rsi_value:.2f}, an elevated reading. "
                    "This rule treats it as neutral rather than assuming "
                    "additional upside."
                ),
            )
        )
    elif rsi_value >= 55:
        signals.append(
            BiasSignal(
                name="rsi14",
                value=rsi_value,
                unit="index",
                direction="bullish",
                explanation=f"RSI is {rsi_value:.2f}, indicating positive momentum.",
            )
        )
    elif rsi_value <= 45:
        signals.append(
            BiasSignal(
                name="rsi14",
                value=rsi_value,
                unit="index",
                direction="bearish",
                explanation=f"RSI is {rsi_value:.2f}, indicating negative momentum.",
            )
        )
    else:
        signals.append(
            BiasSignal(
                name="rsi14",
                value=rsi_value,
                unit="index",
                direction="neutral",
                explanation=(
                    f"RSI is {rsi_value:.2f}, within the neutral momentum zone."
                ),
            )
        )

    macd_histogram = _value_or_none(macd["MACD_histogram"])

    if macd_histogram is None:
        signals.append(
            _unavailable_signal(
                name="macd_histogram",
                unit="price_points",
                explanation="Insufficient history to calculate the MACD histogram.",
            )
        )
    elif macd_histogram > 0:
        signals.append(
            BiasSignal(
                name="macd_histogram",
                value=macd_histogram,
                unit="price_points",
                direction="bullish",
                explanation="The MACD histogram is above zero.",
            )
        )
    elif macd_histogram < 0:
        signals.append(
            BiasSignal(
                name="macd_histogram",
                value=macd_histogram,
                unit="price_points",
                direction="bearish",
                explanation="The MACD histogram is below zero.",
            )
        )
    else:
        signals.append(
            BiasSignal(
                name="macd_histogram",
                value=macd_histogram,
                unit="price_points",
                direction="neutral",
                explanation="The MACD histogram is at zero.",
            )
        )

    lookback = TIMEFRAME_LOOKBACKS[timeframe]
    required_observations = lookback + 1

    if len(prices) < required_observations:
        signals.append(
            _unavailable_signal(
                name="horizon_return",
                unit="percent",
                explanation=(
                    f"The {timeframe} return needs at least "
                    f"{required_observations} daily closing observations; "
                    f"received {len(prices)}."
                ),
            )
        )
    else:
        previous_close = float(prices.iloc[-required_observations])
        return_percent = (latest_close / previous_close - 1.0) * 100.0
        threshold = RETURN_THRESHOLDS_PERCENT[timeframe]

        if return_percent > threshold:
            direction: SignalDirection = "bullish"
            explanation = (
                f"The {timeframe} lookback return is "
                f"+{return_percent:.2f}%, above the +{threshold:.2f}% threshold."
            )
        elif return_percent < -threshold:
            direction = "bearish"
            explanation = (
                f"The {timeframe} lookback return is "
                f"{return_percent:.2f}%, below the -{threshold:.2f}% threshold."
            )
        else:
            direction = "neutral"
            explanation = (
                f"The {timeframe} lookback return is {return_percent:.2f}%, "
                f"within the ±{threshold:.2f}% threshold."
            )

        signals.append(
            BiasSignal(
                name="horizon_return",
                value=return_percent,
                unit="percent",
                direction=direction,
                explanation=explanation,
            )
        )

    score = sum(
        1
        if signal.direction == "bullish"
        else -1
        if signal.direction == "bearish"
        else 0
        for signal in signals
    )

    available_signals = sum(signal.direction != "unavailable" for signal in signals)

    if available_signals < MIN_DIRECTIONAL_SIGNALS:
        bias: BiasDirection = "neutral"
        explanation = (
            f"Neutral bias: only {available_signals} signals are available. "
            f"At least {MIN_DIRECTIONAL_SIGNALS} are required for a directional "
            "bias. Obtain more historical observations and recalculate."
        )
    elif score >= 2:
        bias = "bullish"
        explanation = (
            f"Bullish bias: the available signals have a net score of "
            f"+{score}. This is a rule-based technical assessment."
        )
    elif score <= -2:
        bias = "bearish"
        explanation = (
            f"Bearish bias: the available signals have a net score of "
            f"{score}. This is a rule-based technical assessment."
        )
    else:
        bias = "neutral"
        explanation = (
            f"Neutral bias: the available signals have a net score of "
            f"{score}, which does not meet the directional threshold."
        )

    return MarketBiasResult(
        timeframe=timeframe,
        as_of=pd.Timestamp(prices.index[-1]),
        latest_close=latest_close,
        observations=len(prices),
        bias=bias,
        score=score,
        explanation=explanation,
        signals=tuple(signals),
    )
