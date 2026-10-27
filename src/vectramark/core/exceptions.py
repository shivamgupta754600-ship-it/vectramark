class VectraMarkError(Exception):
    """Base exception for VectraMark."""


class ConfigurationError(VectraMarkError):
    """Raised when application configuration is invalid."""


class MarketDataError(VectraMarkError):
    """Raised when market data cannot be retrieved or processed."""


class DataValidationError(VectraMarkError):
    """Raised when market data fails validation."""


class ModelError(VectraMarkError):
    """Raised when a model operation fails."""


class ResearchError(VectraMarkError):
    """Raised when AI research fails."""


class BacktestError(VectraMarkError):
    """Raised when a backtesting operation fails."""
