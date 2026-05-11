"""Build the POST /poll request payload from the local measurement service's
/state response and agent-side derived values."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

from .config import AgentConfig
from .local_api import LocalApiClient, LocalApiError
from .measurement_translation import translate_measurement

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_calibration_fresh(state: dict[str, Any], max_age_seconds: float) -> bool:
    """Calibration is considered valid if one is loaded AND it's younger than
    the operator-configured staleness threshold."""
    if not state.get("has_calibration", False):
        return False
    age = state.get("calibration_age_seconds")
    if age is None:
        # Local service doesn't expose age yet — fall back to presence only.
        return True
    return age <= max_age_seconds


def _resource_payload() -> dict[str, Any]:
    """Lightweight host-level diagnostics included in every poll's `payload`."""
    disk = shutil.disk_usage("/")
    return {
        "disk_used_pct": round((disk.used / disk.total) * 100, 1) if disk.total else None,
        "disk_free_bytes": disk.free,
    }


def build_poll_payload(
    local: LocalApiClient,
    agent_config: AgentConfig,
    config_hash: Optional[str],
    active_alerts: list[dict[str, Any]],
    operator_thresholds: dict[str, Any],
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Build the /poll request body.

    Returns (poll_payload, local_state). The local_state is returned so the
    caller can reuse it for alert evaluation without a second round-trip.
    """
    try:
        state = local.state()
    except LocalApiError as e:
        logger.warning("local /state failed, sending degraded poll: %s", e)
        state = {}

    max_age = float(
        operator_thresholds.get(
            "calibration_max_age_seconds",
            agent_config.alert_defaults.calibration_max_age_seconds,
        )
    )

    loop_stats = state.get("loop_stats") or {}
    measurement_loop_running = bool(loop_stats.get("is_running", False))

    payload: dict[str, Any] = {
        # Always true — if we can make the call, we're online.
        "online": True,
        "sensor_connected": bool(state.get("sensor_connected", False)),
        "streaming": bool(state.get("streaming", False)),
        "has_valid_calibration": _is_calibration_fresh(state, max_age),
        "measurement_loop_running": measurement_loop_running,
        # Phase 3 — hardcoded False until WITS lands.
        "wits_output_active": False,
        "config_hash": config_hash,
        "recorded_at": _utcnow_iso(),
        "payload": _resource_payload(),
        "active_alerts": active_alerts,
    }

    if agent_config.poll.attach_latest_measurement:
        translated = translate_measurement(state.get("last_measurement"))
        if translated is not None:
            payload["latest_measurement"] = translated

    return payload, state
