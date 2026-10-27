"""Domain logic for historical support and resistance detection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupportResistanceLevel:
    """A price zone formed by nearby historical swing points."""

    price: float
    zone_low: float
    zone_high: float
    kind: str
    touch_count: int
    explanation: str


@dataclass(frozen=True)
class SupportResistanceResult:
    """Support/resistance analysis for a historical OHLC dataset."""

    latest_close: float
    observations: int
    levels: list[SupportResistanceLevel]


def detect_support_resistance(
    data: pd.DataFrame,
    swing_window: int = 3,
    tolerance_pct: float = 0.5,
    max_levels: int = 10,
) -> SupportResistanceResult:
    """Detect swing points and cluster nearby prices into zones."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Market data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Market data cannot be empty.")

    if swing_window < 1 or swing_window > 20:
        raise ValueError("swing_window must be between 1 and 20.")

    if not np.isfinite(tolerance_pct) or not 0.0 <= tolerance_pct <= 5.0:
        raise ValueError("tolerance_pct must be between 0 and 5.")

    if max_levels < 1 or max_levels > 50:
        raise ValueError("max_levels must be between 1 and 50.")

    normalized = data.copy()
    normalized.columns = [str(column).strip().title() for column in normalized.columns]

    required = {"High", "Low", "Close"}
    missing = required.difference(normalized.columns)

    if missing:
        raise ValueError(
            f"Market data is missing required columns: {', '.join(sorted(missing))}."
        )

    prices = normalized[["High", "Low", "Close"]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    values = prices.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Market data contains missing or non-finite prices.")

    if (values <= 0).any():
        raise ValueError("All market prices must be greater than zero.")

    if (prices["High"] < prices["Low"]).any():
        raise ValueError("High prices cannot be lower than Low prices.")

    latest_close = float(prices["Close"].iloc[-1])
    observations = len(prices)
    highs = prices["High"].to_numpy(dtype=float)
    lows = prices["Low"].to_numpy(dtype=float)

    # A swing point must be an extreme within a window on both sides.
    candidates: list[tuple[float, str]] = []

    for index in range(swing_window, observations - swing_window):
        left = index - swing_window
        right = index + swing_window + 1

        high_window = highs[left:right]
        low_window = lows[left:right]

        if highs[index] >= high_window.max() and (
            np.count_nonzero(high_window == highs[index]) == 1
        ):
            candidates.append((float(highs[index]), "high"))

        if lows[index] <= low_window.min() and (
            np.count_nonzero(low_window == lows[index]) == 1
        ):
            candidates.append((float(lows[index]), "low"))

    # Cluster nearby swing prices. Each swing point contributes one touch.
    candidates.sort(key=lambda candidate: candidate[0])
    clusters: list[list[tuple[float, str]]] = []

    tolerance = tolerance_pct / 100.0

    for candidate in candidates:
        price = candidate[0]

        if not clusters:
            clusters.append([candidate])
            continue

        current_cluster = clusters[-1]
        cluster_mean = float(np.mean([item[0] for item in current_cluster]))

        relative_distance = abs(price - cluster_mean) / cluster_mean

        if relative_distance <= tolerance:
            current_cluster.append(candidate)
        else:
            clusters.append([candidate])

    levels: list[SupportResistanceLevel] = []

    for cluster in clusters:
        cluster_prices = [item[0] for item in cluster]
        average_price = float(np.mean(cluster_prices))
        zone_low = float(min(cluster_prices))
        zone_high = float(max(cluster_prices))
        kind = "support" if average_price < latest_close else "resistance"

        if np.isclose(average_price, latest_close, rtol=1e-9, atol=1e-9):
            kind = "near_price"

        touches = len(cluster)
        explanation = (
            f"This {kind.replace('_', ' ')} zone contains {touches} "
            f"historical swing point(s). It is an estimate from past prices, "
            f"not a guaranteed future reaction."
        )

        levels.append(
            SupportResistanceLevel(
                price=round(average_price, 8),
                zone_low=round(zone_low, 8),
                zone_high=round(zone_high, 8),
                kind=kind,
                touch_count=touches,
                explanation=explanation,
            )
        )

    # Prefer zones with more historical swing points; use distance as a tie-break.
    levels.sort(
        key=lambda level: (
            -level.touch_count,
            abs(level.price - latest_close),
        )
    )

    return SupportResistanceResult(
        latest_close=latest_close,
        observations=observations,
        levels=levels[:max_levels],
    )
