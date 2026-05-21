"""Translate local measurement-service measurement objects into the OPS schema.

The local service's measurement dict (from /measurements/latest, /state's
last_measurement, or /measurements/history items) uses field names that don't
all line up with what OPS expects on /poll's latest_measurement, /measurements,
or /measurement-history. This module owns the mapping in one place.

OPS schema fields (per Section 5 of the integration spec):
    volume_m3, volume_litres, fill_ratio, surface_area_m2,
    avg_height_m, max_height_m, valid_pixel_ratio,
    notes, measured_at, payload
"""
from __future__ import annotations

from typing import Any, Optional


# Mapping from local field names to OPS field names. Update this as the
# local measurement schema evolves.
_FIELD_MAP: dict[str, str] = {
    # Local dataclass field           → OPS schema field
    "timestamp_utc":                    "measured_at",
    "estimated_volume_m3":              "volume_m3",
    "estimated_volume_liters":          "volume_litres",
    "relative_fill_ratio":              "fill_ratio",
    "occupied_surface_area_m2":         "surface_area_m2",
    "average_fill_height_m":            "avg_height_m",
    "max_fill_height_m":                "max_height_m",
    "valid_pixel_ratio":                "valid_pixel_ratio",
    "notes":                            "notes",
}

# Top-level OPS fields. Anything in the local measurement that isn't one of
# these (and isn't in _FIELD_MAP) gets bundled into `payload` so nothing is lost.
_OPS_FIELDS = {
    "volume_m3",
    "volume_litres",
    "fill_ratio",
    "surface_area_m2",
    "avg_height_m",
    "max_height_m",
    "valid_pixel_ratio",
    "notes",
    "measured_at",
    "payload",
}


def translate_measurement(local_m: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Convert a local measurement dict to the OPS schema. Returns None if the
    input is None."""
    if local_m is None:
        return None

    out: dict[str, Any] = {}
    leftover: dict[str, Any] = {}

    for key, value in local_m.items():
        if key in _FIELD_MAP:
            out[_FIELD_MAP[key]] = value
        elif key in _OPS_FIELDS:
            # Already-OPS-named field that we didn't list in _FIELD_MAP.
            out[key] = value
        else:
            leftover[key] = value

    # Merge any pre-existing 'payload' from the local measurement with anything
    # we couldn't classify. Leftover fields go under payload so diagnostic
    # info isn't dropped on the floor.
    existing_payload = local_m.get("payload")
    if isinstance(existing_payload, dict):
        # Don't double-include — strip it from leftover.
        leftover.pop("payload", None)
        merged = {**existing_payload, **leftover}
    else:
        merged = leftover

    if merged:
        out["payload"] = merged

    # OPS wants `notes` as a single string; we produce a list of diagnostic flags.
    if isinstance(out.get("notes"), list):
        notes_list = out["notes"]
        out["notes"] = "; ".join(notes_list) if notes_list else None

    return out
