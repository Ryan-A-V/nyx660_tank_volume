"""Evaluate which OPS alerts are currently active from the latest local state.

OPS reconciles open/close by diffing the active_alerts array on each /poll
against its own open-alert table:
  - Alert types reported but not open → opened, operators notified.
  - Alert types currently open but not reported → resolved.
  - Identical → no-op.

So the agent's job is to emit, on every poll, the complete set of conditions
that are currently true. There's no separate "open" or "close" call.
"""
from __future__ import annotations

import logging
import shutil
from typing import Any, Optional

from .config import AgentConfig

logger = logging.getLogger(__name__)


# Disk pressure threshold for jetson_resource_warning. Hardcoded for now —
# we could move into config later if it needs tuning per-unit.
_DISK_USED_PCT_WARN = 90.0


def _calibration_age_seconds(state: dict[str, Any]) -> Optional[float]:
    return state.get("calibration_age_seconds")


def compute_active_alerts(
    state: dict[str, Any],
    agent_config: AgentConfig,
    operator_thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the list of currently-active alerts from the local /state snapshot.

    operator_thresholds is the cached OPS config payload (or {} if none yet).
    Missing fields fall back to agent_config.alert_defaults.
    """
    defaults = agent_config.alert_defaults

    def threshold(key: str, fallback: float) -> float:
        v = operator_thresholds.get(key)
        return float(v) if v is not None else float(fallback)

    tank_full = threshold("tank_full_threshold", defaults.tank_full_threshold)
    tank_low = threshold("tank_low_threshold", defaults.tank_low_threshold)
    min_valid = threshold("min_valid_pixel_ratio", defaults.min_valid_pixel_ratio)
    cal_max_age = threshold(
        "calibration_max_age_seconds", defaults.calibration_max_age_seconds
    )

    alerts: list[dict[str, Any]] = []

    # sensor_disconnected
    if not state.get("sensor_connected", False):
        alerts.append({"alert_type": "sensor_disconnected", "severity": "critical"})

    # measurement_loop_stopped
    loop_stats = state.get("loop_stats") or {}
    if not loop_stats.get("is_running", False):
        alerts.append({"alert_type": "measurement_loop_stopped", "severity": "critical"})

    # calibration_missing_or_stale
    has_cal = state.get("has_calibration", False)
    age = _calibration_age_seconds(state)
    if not has_cal:
        alerts.append(
            {"alert_type": "calibration_missing_or_stale", "severity": "warning"}
        )
    elif age is not None and age > cal_max_age:
        alerts.append(
            {
                "alert_type": "calibration_missing_or_stale",
                "severity": "warning",
                "payload": {"calibration_age_seconds": age},
            }
        )

    # tank_full / tank_low / low_signal_quality — derived from latest measurement.
    latest = state.get("last_measurement") or {}
    fill_ratio = latest.get("fill_ratio")
    if isinstance(fill_ratio, (int, float)):
        if fill_ratio >= tank_full:
            alerts.append(
                {
                    "alert_type": "tank_full",
                    "severity": "warning",
                    "payload": {"fill_ratio": fill_ratio},
                }
            )
        elif fill_ratio <= tank_low:
            alerts.append(
                {
                    "alert_type": "tank_low",
                    "severity": "warning",
                    "payload": {"fill_ratio": fill_ratio},
                }
            )

    valid_ratio = latest.get("valid_pixel_ratio")
    if isinstance(valid_ratio, (int, float)) and valid_ratio < min_valid:
        alerts.append(
            {
                "alert_type": "low_signal_quality",
                "severity": "warning",
                "payload": {"valid_pixel_ratio": valid_ratio},
            }
        )

    # jetson_resource_warning — host-side disk pressure.
    try:
        disk = shutil.disk_usage("/")
        used_pct = (disk.used / disk.total) * 100 if disk.total else 0.0
        if used_pct >= _DISK_USED_PCT_WARN:
            alerts.append(
                {
                    "alert_type": "jetson_resource_warning",
                    "severity": "warning",
                    "payload": {"disk_used_pct": round(used_pct, 1)},
                }
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("disk check failed: %s", e)

    # wits_output_failure — phase 3, not evaluated yet.

    return alerts
