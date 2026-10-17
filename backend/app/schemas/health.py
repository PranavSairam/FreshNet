from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    available: bool
    status: str = Field(description="ready or unavailable. The API can be up while the model is not.")
    message: str
    architecture: str
    model_version: str = ""


class DatabaseStatus(BaseModel):
    ok: bool
    path: str
    message: str


class ProduceModelStatus(BaseModel):
    available: bool
    status: str = Field(description="ready or unavailable. When unavailable, produce type is chosen manually.")
    model_version: str = ""
    confirmation_threshold: float | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model: ModelStatus
    produce_model: ProduceModelStatus
    database: DatabaseStatus
