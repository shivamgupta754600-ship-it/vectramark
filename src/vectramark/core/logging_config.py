import logging
from logging.handlers import RotatingFileHandler

from vectramark.config.settings import get_settings


def setup_logging() -> None:
    """Configure console and rotating-file logging."""
    settings = get_settings()

    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.logs_dir / "vectramark.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for a module."""
    return logging.getLogger(name)
