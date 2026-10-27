from contextlib import asynccontextmanager

from fastapi import FastAPI

from vectramark.api.routes.analytics import router as analytics_router
from vectramark.api.routes.backtesting import router as backtesting_router
from vectramark.api.routes.forecasting import router as forecasting_router
from vectramark.api.routes.health import router as health_router
from vectramark.api.routes.market_data import router as market_data_router
from vectramark.config.settings import get_settings
from vectramark.core.logging_config import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure application resources during startup and shutdown."""
    setup_logging()

    logger = get_logger(__name__)
    logger.info("VectraMark backend starting")

    yield

    logger.info("VectraMark backend shutting down")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=("VectraMark market intelligence and predictive analytics API."),
    lifespan=lifespan,
)

app.include_router(market_data_router)
app.include_router(analytics_router)
app.include_router(forecasting_router)
app.include_router(health_router)
app.include_router(backtesting_router)


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }
