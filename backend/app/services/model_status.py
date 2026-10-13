from app.core.config import Settings
from app.schemas.health import ModelStatus
from app.services.runtime import current_model_status

ARCHITECTURE = "fused"


def get_model_status(_settings: Settings) -> ModelStatus:
    return current_model_status()
