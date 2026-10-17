from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.db.sqlite import ping_database
from app.schemas.health import DatabaseStatus, HealthResponse, ProduceModelStatus
from app.services.model_status import get_model_status
from app.services.produce import automatic_identification_info

logger = logging.getLogger("freshnet.health")
router = APIRouter()


def _produce_status() -> ProduceModelStatus:
    info = automatic_identification_info()
    if not info["available"]:
        return ProduceModelStatus(available=False, status="unavailable")
    return ProduceModelStatus(
        available=True,
        status="ready",
        model_version=str(info["model_version"] or ""),
        confirmation_threshold=info["confirmation_threshold"],
    )


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    model = get_model_status(settings)
    try:
        ping_database(settings.sqlite_file)
        database = DatabaseStatus(
            ok=True,
            path="local-sqlite",
            message="SQLite reachable",
        )
    except Exception:
        logger.exception("SQLite health check failed")
        database = DatabaseStatus(
            ok=False,
            path="local-sqlite",
            message="SQLite unreachable",
        )

    api_status = "ok" if database.ok else "degraded"
    logger.info(
        "Health check",
        extra={
            "api_status": api_status,
            "model_status": model.status,
            "model_available": model.available,
            "database_ok": database.ok,
        },
    )
    return HealthResponse(
        status=api_status,
        service=settings.app_name,
        version=settings.app_version,
        model=model,
        produce_model=_produce_status(),
        database=database,
    )
