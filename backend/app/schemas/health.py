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


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model: ModelStatus
    database: DatabaseStatus
