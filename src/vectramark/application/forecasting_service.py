"""Application service for baseline market-price forecasting."""

from dataclasses import dataclass

import pandas as pd

from vectramark.application.market_data_service import MarketDataService
from vectramark.domain.forecasting.baseline import naive_forecast

MAX_FORECAST_HORIZON = 100

INTRADAY_INTERVAL_MINUTES = {
    "1m": 1,
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "90m": 90,
    "1h": 60,
}

SUPPORTED_INTERVALS = frozenset(
    {
        *INTRADAY_INTERVAL_MINUTES,
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }
)


@dataclass(frozen=True)
class ForecastPoint:
    """One forecasted price point."""

    step: int
    timestamp: pd.Timestamp
    predicted_close: float


@dataclass(frozen=True)
class ForecastResult:
    """Result returned by the forecasting application service."""

    symbol: str
    period: str
    interval: str
    horizon: int
    latest_observed_at: pd.Timestamp
    latest_observed_close: float
    points: tuple[ForecastPoint, ...]


class ForecastingService:
    """Coordinate market-data retrieval and baseline forecasting."""

    def __init__(self, market_data_service: MarketDataService) -> None:
        self._market_data_service = market_data_service

    def forecast(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        horizon: int = 5,
    ) -> ForecastResult:
        """Generate a baseline forecast from historical closing prices."""

        if not isinstance(symbol, str):
            raise TypeError("Market symbol must be a string.")

        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("Market symbol cannot be empty.")

        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an integer.")

        if not 1 <= horizon <= MAX_FORECAST_HORIZON:
            raise ValueError(f"horizon must be between 1 and {MAX_FORECAST_HORIZON}.")

        if not isinstance(period, str) or not period.strip():
            raise ValueError("period must be a non-empty string.")

        if interval not in SUPPORTED_INTERVALS:
            raise ValueError("Unsupported forecast interval.")

        market_data = self._market_data_service.fetch_and_store(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
        )

        if not isinstance(market_data, pd.DataFrame):
            raise ValueError("Market-data service returned invalid data.")

        if market_data.empty:
            raise ValueError("Historical market data cannot be empty.")

        if "Close" not in market_data.columns:
            raise ValueError("Historical market data must contain a Close column.")

        if not isinstance(market_data.index, pd.DatetimeIndex):
            raise ValueError("Market-data index must contain timestamps.")

        closing_prices = pd.to_numeric(
            market_data["Close"].copy(),
            errors="raise",
        )

        forecast_prices = naive_forecast(
            closing_prices,
            horizon=horizon,
        )

        latest_timestamp = pd.Timestamp(closing_prices.index[-1])
        future_timestamps = self._make_future_timestamps(
            latest_timestamp=latest_timestamp,
            horizon=horizon,
            interval=interval,
        )

        points = tuple(
            ForecastPoint(
                step=step,
                timestamp=future_timestamps[step - 1],
                predicted_close=float(forecast_prices.iloc[step - 1]),
            )
            for step in range(1, horizon + 1)
        )

        return ForecastResult(
            symbol=normalized_symbol,
            period=period,
            interval=interval,
            horizon=horizon,
            latest_observed_at=latest_timestamp,
            latest_observed_close=float(closing_prices.iloc[-1]),
            points=points,
        )

    @staticmethod
    def _make_future_timestamps(
        latest_timestamp: pd.Timestamp,
        horizon: int,
        interval: str,
    ) -> tuple[pd.Timestamp, ...]:
        """Estimate future timestamps from the requested interval.

        These timestamps are interval-based estimates, not an exchange
        trading calendar. Trading sessions, holidays, and market closures
        are not modelled.
        """

        timestamps = []

        for step in range(1, horizon + 1):
            if interval in INTRADAY_INTERVAL_MINUTES:
                minutes = INTRADAY_INTERVAL_MINUTES[interval]
                timestamp = latest_timestamp + pd.Timedelta(minutes=minutes * step)
            elif interval == "1d":
                timestamp = latest_timestamp + pd.offsets.BDay(step)
            elif interval == "5d":
                timestamp = latest_timestamp + pd.offsets.BDay(5 * step)
            elif interval == "1wk":
                timestamp = latest_timestamp + pd.DateOffset(weeks=step)
            elif interval == "1mo":
                timestamp = latest_timestamp + pd.DateOffset(months=step)
            elif interval == "3mo":
                timestamp = latest_timestamp + pd.DateOffset(months=3 * step)
            else:
                raise ValueError("Unsupported forecast interval.")

            timestamps.append(pd.Timestamp(timestamp))

        return tuple(timestamps)
