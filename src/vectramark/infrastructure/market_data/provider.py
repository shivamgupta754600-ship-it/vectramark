from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    """Abstract interface for historical market data providers."""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return historical OHLCV data for a market symbol."""
        raise NotImplementedError
