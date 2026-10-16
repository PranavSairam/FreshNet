from __future__ import annotations

from fastapi import APIRouter

from app.services.produce import (
    automatic_identification_info,
    catalogue_version,
    describe_result,
    list_produce_options,
    parse_selected_produce,
)

router = APIRouter()


@router.get("/produce")
def produce_options() -> dict:
    """Supported produce types for the selector, and whether automatic identification exists."""
    return {
        "catalogue_version": catalogue_version(),
        "options": list_produce_options(),
        "identification": {
            "mode": "user_selected",
            **automatic_identification_info(),
        },
    }


@router.get("/guidance")
def guidance(produce: str = "unknown", freshness: str = "Fresh") -> dict:
    """Reviewed guidance for a produce type and freshness class.

    Used when the user changes the produce type after a result. It reads only the
    catalogue. The image is not reclassified and the model is not involved.
    """
    identity = parse_selected_produce(produce)
    return describe_result(freshness, identity)
