"""Supported-image checks for the HTTP API.

These checks run before the frozen bundle preprocessing. They do not rotate,
composite, or resize the image.
"""

from __future__ import annotations

import io

from PIL import Image

from app.core.errors import AppError

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_EDGE_PIXELS = 8_000
MAX_PIXELS = 16_000_000


def validate_image_bytes(data: bytes) -> Image.Image:
    if not data:
        raise AppError("empty_upload", "Choose a JPEG or PNG image.", 400)
    if len(data) > MAX_UPLOAD_BYTES:
        raise AppError(
            "upload_too_large",
            "That file is larger than 8 MB. Export a smaller JPEG or PNG.",
            413,
        )
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Image.DecompressionBombError as exc:
            raise AppError(
                "image_too_large",
                "That image has too many pixels. Export a smaller JPEG or PNG.",
                413,
            ) from exc
        except Exception as exc:
            raise AppError(
                "invalid_image",
                "That file could not be decoded as a JPEG or PNG.",
                400,
            ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if image.format not in {"JPEG", "PNG"}:
        raise AppError(
            "unsupported_format",
            "Use a static JPEG or PNG. Other formats are not accepted.",
            415,
        )
    frames = int(getattr(image, "n_frames", 1) or 1)
    if getattr(image, "is_animated", False) or frames > 1:
        raise AppError(
            "animated_image",
            "Animated images are not accepted. Export one still JPEG or PNG frame.",
            415,
        )
    if _has_transparency(image):
        raise AppError(
            "transparency_unsupported",
            "Images with transparency are not accepted. Export a JPEG or PNG without an alpha channel.",
            415,
        )
    orientation = _exif_orientation(image)
    if orientation not in (None, 1):
        raise AppError(
            "exif_orientation",
            "This image stores rotation in EXIF instead of its pixels. Export an upright JPEG or PNG and upload that file.",
            415,
        )
    width, height = image.size
    if width < 1 or height < 1 or width > MAX_EDGE_PIXELS or height > MAX_EDGE_PIXELS or width * height > MAX_PIXELS:
        raise AppError(
            "image_too_large",
            "That image has too many pixels. Export a smaller JPEG or PNG.",
            413,
        )
    if image.mode == "RGB":
        return image
    if image.mode in {"L", "P"}:
        return image.convert("RGB")
    raise AppError(
        "unsupported_format",
        "Use a static JPEG or PNG in RGB or grayscale.",
        415,
    )


def _has_transparency(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def _exif_orientation(image: Image.Image) -> int | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    value = exif.get(274)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
