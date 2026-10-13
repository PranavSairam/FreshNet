from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ErrorBody(BaseModel):
    code: str
    message: str


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        body = ErrorBody(code=exc.code, message=exc.message)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, _exc: Exception) -> JSONResponse:
        body = ErrorBody(
            code="internal_error",
            message="An unexpected server error occurred.",
        )
        return JSONResponse(status_code=500, content=body.model_dump())
