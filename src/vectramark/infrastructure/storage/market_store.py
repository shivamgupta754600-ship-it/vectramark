import os
import tempfile
from pathlib import Path

import pandas as pd

from vectramark.config.settings import get_settings
from vectramark.core.exceptions import MarketDataError
from vectramark.domain.market_data.validator import MarketDataValidator

SUPPORTED_SYMBOLS = frozenset(
    {
        "XAUUSD",
        "XAGUSD",
        "NASDAQ100",
        "SP500",
        "DJIA",
    }
)


class LocalMarketStore:
    """Persist validated market data in a local directory."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        validator: MarketDataValidator | None = None,
    ) -> None:
        configured_dir = (
            data_dir if data_dir is not None else get_settings().processed_data_dir
        )

        self._data_dir = Path(configured_dir).expanduser().resolve()
        self._validator = validator or MarketDataValidator()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if not isinstance(symbol, str):
            raise MarketDataError("Market symbol must be a string.")

        normalized = symbol.strip().upper()

        if normalized not in SUPPORTED_SYMBOLS:
            raise MarketDataError("Unsupported market symbol.")

        return normalized

    def _path_for(self, symbol: str) -> Path:
        normalized = self._normalize_symbol(symbol)
        path = (self._data_dir / f"{normalized}.csv").resolve()

        # Defense in depth: the generated file must remain in our store.
        if path.parent != self._data_dir:
            raise MarketDataError("Invalid market data storage path.")

        return path

    def save(self, symbol: str, data: pd.DataFrame) -> Path:
        """Validate and atomically save a market dataset."""
        path = self._path_for(symbol)
        validated = self._validator.validate(data)

        temporary_path: Path | None = None

        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                suffix=".tmp",
                prefix=f".{path.stem}_",
                dir=self._data_dir,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                validated.to_csv(
                    temporary_file,
                    index=True,
                    index_label="timestamp",
                )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, path)
            temporary_path = None

            return path

        except (OSError, ValueError) as exc:
            raise MarketDataError(f"Could not save market data for {symbol}.") from exc

        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    # Preserve the original failure; a leftover temporary
                    # file can be cleaned up by a maintenance task.
                    pass

    def load(self, symbol: str) -> pd.DataFrame:
        """Load and validate a previously saved market dataset."""
        path = self._path_for(symbol)

        try:
            if not path.is_file():
                raise MarketDataError(f"No saved market data exists for {symbol}.")

            data = pd.read_csv(
                path,
                parse_dates=["timestamp"],
                index_col="timestamp",
            )

            return self._validator.validate(data)

        except MarketDataError:
            raise

        except (
            OSError,
            ValueError,
            KeyError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as exc:
            raise MarketDataError(
                f"Could not load valid market data for {symbol}."
            ) from exc
