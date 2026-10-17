from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.produce import router as produce_router
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.db.sqlite import initialize_database
from app.services.produce import load_catalogue, set_identifier
from app.services.produce_model import load_produce_identifier
from app.services.runtime import load_model

logger = logging.getLogger("freshnet")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    initialize_database(settings.sqlite_file)
    load_catalogue()  # fail fast if the guidance catalogue is malformed
    load_model(settings)
    set_identifier(load_produce_identifier(settings))  # None -> manual selection only
    logger.info("FreshNet API started", extra={"environment": settings.environment})
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        summary="Local API for visual food freshness estimates.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(application)
    application.include_router(health_router, prefix="/api", tags=["health"])
    application.include_router(predict_router, prefix="/api", tags=["predict"])
    application.include_router(produce_router, prefix="/api", tags=["produce"])

    @application.get("/", tags=["health"])
    def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "health": "/api/health",
            "docs": "/docs",
        }

    return application


app = create_app()
