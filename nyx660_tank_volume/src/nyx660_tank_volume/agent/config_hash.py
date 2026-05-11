"""Canonical SHA-256 of a config file.

OPS computes config_hash as SHA-256 over a canonicalized JSON form of the
config: keys recursively sorted lexicographically, no whitespace. The unit
must compute the same hash over its currently-running config so OPS can do
a hash-based delta on every /poll.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _canonicalize(obj: Any) -> Any:
    """Recursively sort dict keys. Arrays are left as-is per the OPS spec."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(item) for item in obj]
    return obj


def compute_config_hash(config_path: Path) -> str:
    """SHA-256 hex over the canonicalized config JSON."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    canonical = _canonicalize(raw)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_dict_hash(obj: dict[str, Any]) -> str:
    """Same canonical hash but over an in-memory dict (e.g. a config payload
    OPS just sent us that we haven't written to disk yet)."""
    canonical = _canonicalize(obj)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
