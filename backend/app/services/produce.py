"""Produce identity, combined labels and reviewed guidance.

Everything here is deterministic and independent of the freshness model:

* Produce identity is either chosen by the user from a fixed catalogue, produced
  by an optional separate classifier (none ships today), or unknown.
* Guidance is looked up from a versioned, reviewed catalogue keyed by
  produce type and predicted freshness class. No text supplied by a client is
  ever placed into guidance.
* Spoilage severity is never estimated. The current model only separates
  Fresh from Spoiled.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.core.errors import AppError

logger = logging.getLogger("freshnet.produce")

CATALOGUE_PATH = Path(__file__).resolve().parents[1] / "data" / "guidance_catalogue.json"
FRESHNESS_LABELS = ("Fresh", "Spoiled")
UNKNOWN_ID = "unknown"
UNKNOWN_LABEL = "Unknown / other"
SOURCE_USER = "user_selected"
SOURCE_MODEL = "model_predicted"
SOURCE_UNKNOWN = "unknown"
MAX_PRODUCE_FIELD_LENGTH = 40
_ID_PATTERN = re.compile(r"^[a-z_]+$")
_REVIEW_STATES = {"reviewed", "awaiting_review"}

SEVERITY_EXPLANATION = (
    "The current model classifies Fresh or Spoiled but does not measure the degree of spoilage."
)
GUIDANCE_DISCLAIMER = (
    "General food-handling advice. It is not based on what the model saw in your photo."
)


# --------------------------------------------------------------------------- catalogue


class CatalogueError(RuntimeError):
    """The guidance catalogue is malformed. Raised at load time, never per request."""


@lru_cache(maxsize=1)
def load_catalogue() -> dict[str, Any]:
    with CATALOGUE_PATH.open(encoding="utf-8") as handle:
        catalogue = json.load(handle)
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue(catalogue: dict[str, Any]) -> None:
    for key in ("catalogue_version", "reviewed_on", "safety_note", "sources", "shared", "produce"):
        if key not in catalogue:
            raise CatalogueError(f"missing catalogue key: {key}")
    sources = catalogue["sources"]
    if not sources:
        raise CatalogueError("catalogue has no sources")
    for source_id, source in sources.items():
        for key in ("title", "publisher", "url", "scope", "reviewed_on"):
            if not source.get(key):
                raise CatalogueError(f"source {source_id} missing {key}")
        if not str(source["url"]).startswith("https://"):
            raise CatalogueError(f"source {source_id} url must be https")

    def check_points(points: list[dict[str, Any]], where: str) -> None:
        for point in points:
            if not point.get("text"):
                raise CatalogueError(f"{where}: empty point")
            if not point.get("sources"):
                raise CatalogueError(f"{where}: point has no source")
            for source_id in point["sources"]:
                if source_id not in sources:
                    raise CatalogueError(f"{where}: unknown source {source_id}")

    shared = catalogue["shared"]
    for label in FRESHNESS_LABELS:
        if label not in shared:
            raise CatalogueError(f"shared guidance missing {label}")
    check_points(shared["Fresh"]["next_steps"], "shared Fresh next_steps")
    check_points(shared["Fresh"]["storage_fallback"], "shared Fresh storage_fallback")
    check_points(shared["Fresh"]["checks"], "shared Fresh checks")
    for section in ("next_steps", "storage", "checks"):
        check_points(shared["Spoiled"][section], f"shared Spoiled {section}")

    produce = catalogue["produce"]
    if UNKNOWN_ID in produce:
        raise CatalogueError("'unknown' is reserved and cannot be a produce entry")
    for produce_id, entry in produce.items():
        if not _ID_PATTERN.match(produce_id):
            raise CatalogueError(f"bad produce id: {produce_id}")
        for key in ("label", "noun", "storage_review", "storage"):
            if key not in entry:
                raise CatalogueError(f"{produce_id} missing {key}")
        if entry["storage_review"] not in _REVIEW_STATES:
            raise CatalogueError(f"{produce_id} has unknown review state")
        if entry["storage_review"] == "reviewed" and not entry["storage"]:
            raise CatalogueError(f"{produce_id} is marked reviewed but has no storage points")
        if entry["storage_review"] == "awaiting_review" and entry["storage"]:
            raise CatalogueError(f"{produce_id} is awaiting review and must not carry storage points")
        check_points(entry["storage"], f"{produce_id} storage")
        if "spoiled_mold_point" in entry:
            check_points([entry["spoiled_mold_point"]], f"{produce_id} spoiled_mold_point")


def catalogue_version() -> str:
    return str(load_catalogue()["catalogue_version"])


# --------------------------------------------------------------------------- identity


@dataclass(frozen=True)
class ProduceIdentity:
    produce_id: str
    label: str
    source: str
    model_version: str | None = None
    probabilities: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.produce_id,
            "label": self.label,
            "source": self.source,
            "model_version": self.model_version,
            "probabilities": self.probabilities,
        }


def unknown_identity() -> ProduceIdentity:
    return ProduceIdentity(UNKNOWN_ID, UNKNOWN_LABEL, SOURCE_UNKNOWN)


def list_produce_options() -> list[dict[str, str]]:
    produce = load_catalogue()["produce"]
    options = [{"id": pid, "label": entry["label"]} for pid, entry in produce.items()]
    options.sort(key=lambda option: option["label"].lower())
    options.append({"id": UNKNOWN_ID, "label": UNKNOWN_LABEL})
    return options


def parse_selected_produce(value: str | None) -> ProduceIdentity:
    """Turn a client-supplied produce id into an identity.

    Only exact ids from the catalogue are accepted. The raw value is never echoed
    back in an error or placed into any generated text.
    """
    if value is None:
        return unknown_identity()
    if len(value) > MAX_PRODUCE_FIELD_LENGTH:
        raise _unsupported_produce()
    candidate = value.strip()
    if candidate == "" or candidate == UNKNOWN_ID:
        return unknown_identity()
    produce = load_catalogue()["produce"]
    if not _ID_PATTERN.match(candidate) or candidate not in produce:
        raise _unsupported_produce()
    return ProduceIdentity(candidate, produce[candidate]["label"], SOURCE_USER)


def _unsupported_produce() -> AppError:
    return AppError(
        "unsupported_produce",
        "That produce type is not supported. Choose one from the list or leave it as Unknown / other.",
        400,
    )


class ProduceIdentifier(Protocol):
    """Interface for a future automatic produce classifier.

    It must be a separate model with its own version and probabilities. It must not
    reuse the freshness model, and its accuracy must be measured separately.
    No implementation ships with FreshNet today.
    """

    model_version: str

    def identify(self, image: Any) -> ProduceIdentity | None: ...


_identifier: ProduceIdentifier | None = None


def set_identifier(identifier: ProduceIdentifier | None) -> None:
    global _identifier
    _identifier = identifier


def automatic_identification_available() -> bool:
    return _identifier is not None


def automatic_identification_info() -> dict[str, Any]:
    return {
        "available": _identifier is not None,
        "model_version": getattr(_identifier, "model_version", None) if _identifier else None,
    }


def resolve_identity(selected: ProduceIdentity, image: Any) -> ProduceIdentity:
    """Pick the identity for a result.

    A user selection always wins. Otherwise try the optional automatic identifier.
    If it is missing, returns nothing, returns an unsupported id, or raises, identity
    is unknown and the freshness result is unaffected.
    """
    if selected.source == SOURCE_USER:
        return selected
    if _identifier is None:
        return selected
    try:
        predicted = _identifier.identify(image)
    except Exception:
        logger.exception("Automatic produce identification failed; identity left unknown")
        return unknown_identity()
    produce = load_catalogue()["produce"]
    if predicted is None or predicted.produce_id not in produce:
        return unknown_identity()
    return ProduceIdentity(
        predicted.produce_id,
        produce[predicted.produce_id]["label"],
        SOURCE_MODEL,
        predicted.model_version,
        predicted.probabilities,
    )


# --------------------------------------------------------------------------- labels


def combined_label(freshness_label: str, identity: ProduceIdentity) -> str:
    """'Fresh banana', 'Spoiled apple', or 'Fresh produce' when identity is unknown."""
    _require_freshness(freshness_label)
    if identity.produce_id == UNKNOWN_ID:
        noun = "produce"
    else:
        noun = load_catalogue()["produce"][identity.produce_id]["noun"]
    return f"{freshness_label} {noun}"


def severity_block() -> dict[str, Any]:
    return {"status": "not_assessed", "label": None, "explanation": SEVERITY_EXPLANATION}


def _require_freshness(label: str) -> None:
    if label not in FRESHNESS_LABELS:
        raise AppError("unsupported_freshness", "Freshness must be Fresh or Spoiled.", 400)


# --------------------------------------------------------------------------- guidance


def build_guidance(produce_id: str, freshness_label: str) -> dict[str, Any]:
    _require_freshness(freshness_label)
    catalogue = load_catalogue()
    shared = catalogue["shared"][freshness_label]
    entry = catalogue["produce"].get(produce_id) if produce_id != UNKNOWN_ID else None
    if produce_id != UNKNOWN_ID and entry is None:
        raise _unsupported_produce()

    next_steps = [dict(point) for point in shared["next_steps"]]
    if freshness_label == "Fresh":
        if entry is not None and entry["storage_review"] == "reviewed":
            storage = [dict(point) for point in entry["storage"]]
            review = "reviewed"
        else:
            storage = [dict(point) for point in shared["storage_fallback"]]
            review = "awaiting_review" if entry is not None else "generic"
    else:
        storage = [dict(point) for point in shared["storage"]]
        review = "generic" if entry is None else "reviewed"
        if entry is not None and "spoiled_mold_point" in entry:
            next_steps = [
                dict(entry["spoiled_mold_point"]) if point.get("id") == "mold_general" else point
                for point in next_steps
            ]
    checks = [dict(point) for point in shared["checks"]]

    for group in (next_steps, storage, checks):
        for point in group:
            point.pop("id", None)

    used = {sid for group in (next_steps, storage, checks) for point in group for sid in point["sources"]}
    sources = [
        {"id": sid, **source}
        for sid, source in catalogue["sources"].items()
        if sid in used
    ]
    return {
        "catalogue_version": catalogue["catalogue_version"],
        "general_advice": next_steps,
        "storage": storage,
        "checks": checks,
        "review_status": review,
        "safety_note": catalogue["safety_note"],
        "disclaimer": GUIDANCE_DISCLAIMER,
        "sources": sources,
    }


def describe_result(freshness_label: str, identity: ProduceIdentity) -> dict[str, Any]:
    """Produce / combined label / severity / guidance for one freshness result."""
    return {
        "combined_label": combined_label(freshness_label, identity),
        "produce": identity.as_dict(),
        "severity": severity_block(),
        "guidance": build_guidance(identity.produce_id, freshness_label),
    }
