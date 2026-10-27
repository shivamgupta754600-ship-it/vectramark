import pandas as pd

from vectramark.core.exceptions import MarketDataError
from vectramark.data.market_config import get_market
from vectramark.domain.market_data.validator import MarketDataValidator
from vectramark.infrastructure.market_data.provider import (
    MarketDataProvider,
)
from vectramark.infrastructure.storage.market_store import LocalMarketStore


class MarketDataService:
    """Coordinate market data retrieval, validation, and persistence."""

    def __init__(
        self,
        provider: MarketDataProvider,
        store: LocalMarketStore,
        validator: MarketDataValidator | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._validator = validator or MarketDataValidator()

    def fetch_and_store(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch, validate, persist, and return historical market data."""
        if not isinstance(symbol, str):
            raise MarketDataError("Market symbol must be a string.")

        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise MarketDataError("Market symbol cannot be empty.")

        # Confirm that the symbol is supported by VectraMark.
        get_market(normalized_symbol)

        try:
            raw_data = self._provider.get_historical_data(
                symbol=normalized_symbol,
                period=period,
                interval=interval,
            )

            validated_data = self._validator.validate(raw_data)

            # LocalMarketStore validates again before persistence. This is
            # intentional defense in depth at the storage boundary.
            self._store.save(normalized_symbol, validated_data)

            return validated_data.copy()

        except MarketDataError:
            raise

        except Exception as exc:
            raise MarketDataError(
                f"Market data processing failed for {normalized_symbol}."
            ) from exc

    def load_saved_data(self, symbol: str) -> pd.DataFrame:
        """Load previously persisted market data."""
        if not isinstance(symbol, str):
            raise MarketDataError("Market symbol must be a string.")

        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise MarketDataError("Market symbol cannot be empty.")

        get_market(normalized_symbol)

        try:
            return self._store.load(normalized_symbol)

        except MarketDataError:
            raise

        except Exception as exc:
            raise MarketDataError(
                f"Could not retrieve saved data for {normalized_symbol}."
            ) from exc
