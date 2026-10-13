from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, UploadFile

from app.services.runtime import predict_bytes

logger = logging.getLogger("freshnet.predict")
router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    return await asyncio.to_thread(predict_bytes, data)
