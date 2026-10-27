"""Application service for historical forecast evaluation."""

import pandas as pd

from vectramark.application.market_data_service import MarketDataService
from vectramark.core.exceptions import MarketDataError
from vectramark.data.market_config import get_market
from vectramark.domain.forecasting.backtesting import (
    BacktestResult,
    walk_forward_backtest,
)
from vectramark.domain.forecasting.model_comparison import (
    ModelComparisonResult,
    compare_forecasters,
)

MIN_TRAIN_SIZE = 10
MAX_TRAIN_SIZE = 5000
MIN_HORIZON = 1
MAX_HORIZON = 100


class BacktestingService:
    """Coordinate historical data retrieval and walk-forward backtesting."""

    def __init__(self, market_data_service: MarketDataService) -> None:
        self._market_data_service = market_data_service

    def backtest(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        initial_train_size: int = 60,
        horizon: int = 5,
    ) -> BacktestResult:
        """Evaluate baseline forecasts on chronologically later observations."""

        if not isinstance(symbol, str):
            raise TypeError("Market symbol must be a string.")

        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("Market symbol cannot be empty.")

        get_market(normalized_symbol)

        if not isinstance(period, str) or not period.strip():
            raise ValueError("period must be a non-empty string.")

        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("interval must be a non-empty string.")

        if isinstance(initial_train_size, bool) or not isinstance(
            initial_train_size, int
        ):
            raise TypeError("initial_train_size must be an integer.")

        if not MIN_TRAIN_SIZE <= initial_train_size <= MAX_TRAIN_SIZE:
            raise ValueError(
                f"initial_train_size must be between "
                f"{MIN_TRAIN_SIZE} and {MAX_TRAIN_SIZE}."
            )

        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an integer.")

        if not MIN_HORIZON <= horizon <= MAX_HORIZON:
            raise ValueError(
                f"horizon must be between {MIN_HORIZON} and {MAX_HORIZON}."
            )

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

        if len(closing_prices) <= initial_train_size:
            raise ValueError(
                "Insufficient historical observations: the dataset must contain "
                "at least one observation after the initial training window."
            )

        try:
            return walk_forward_backtest(
                closing_prices=closing_prices,
                initial_train_size=initial_train_size,
                horizon=horizon,
            )
        except MarketDataError:
            raise

    def compare_models(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        initial_train_size: int = 60,
        horizon: int = 5,
    ) -> ModelComparisonResult:
        """Compare naive and moving-average forecasts on the same history."""

        if not isinstance(symbol, str):
            raise TypeError("Market symbol must be a string.")

        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("Market symbol cannot be empty.")

        get_market(normalized_symbol)

        if not isinstance(period, str) or not period.strip():
            raise ValueError("period must be a non-empty string.")

        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("interval must be a non-empty string.")

        if isinstance(initial_train_size, bool) or not isinstance(
            initial_train_size, int
        ):
            raise TypeError("initial_train_size must be an integer.")

        if not MIN_TRAIN_SIZE <= initial_train_size <= MAX_TRAIN_SIZE:
            raise ValueError(
                f"initial_train_size must be between "
                f"{MIN_TRAIN_SIZE} and {MAX_TRAIN_SIZE}."
            )

        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an integer.")

        if not MIN_HORIZON <= horizon <= MAX_HORIZON:
            raise ValueError(
                f"horizon must be between {MIN_HORIZON} and {MAX_HORIZON}."
            )

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

        if len(closing_prices) <= initial_train_size:
            raise ValueError(
                "Insufficient historical observations: the dataset must contain "
                "at least one observation after the initial training window."
            )

        return compare_forecasters(
            closing_prices=closing_prices,
            initial_train_size=initial_train_size,
            horizon=horizon,
        )
