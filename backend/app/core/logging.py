from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
        force=True,
    )
    logging.getLogger("freshnet").info(
        "Logging configured",
        extra={"environment": settings.environment},
    )
