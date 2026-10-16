from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, Form, UploadFile

from app.services.produce import parse_selected_produce
from app.services.runtime import predict_bytes

logger = logging.getLogger("freshnet.predict")
router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    produce: str | None = Form(default=None),
) -> dict:
    # Cheap, deterministic validation first: an unsupported produce id is a 400
    # whether or not the model is loaded, and its text is never echoed back.
    selected = parse_selected_produce(produce)
    data = await file.read()
    return await asyncio.to_thread(predict_bytes, data, selected)
