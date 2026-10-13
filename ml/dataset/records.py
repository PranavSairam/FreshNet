from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ImageRecord:
    image_path: str
    label_raw: str
    label: str | None = None
    food_type: str | None = None
    specimen_id: str | None = None
    capture_session: str | None = None
    source: str | None = None
    split_official: str | None = None
    group_id: str | None = None
    width: int | None = None
    height: int | None = None
    readable: bool | None = None
    content_sha256: str | None = None
    phash: str | None = None
    exact_duplicate_of: str | None = None
    near_duplicate_group: str | None = None
    problems: list[str] = field(default_factory=list)
    fixture: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
