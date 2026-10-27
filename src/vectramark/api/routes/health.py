from datetime import UTC, datetime

from fastapi import APIRouter

from vectramark.config.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["Health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""
    settings = get_settings()

    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(UTC).isoformat(),
    }
