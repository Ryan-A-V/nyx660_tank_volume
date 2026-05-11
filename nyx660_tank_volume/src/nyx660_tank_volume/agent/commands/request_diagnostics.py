"""request_diagnostics — gather diagnostics and post via /diagnostics (auto-completes).

This is the pattern for commands that auto-complete server-side via a
dedicated upload endpoint. The handler acknowledges, gathers the data,
then POSTs to /diagnostics which marks the command completed on OPS's side.
We also mark it completed locally to keep state.db consistent.
"""
from __future__ import annotations

import logging
import shutil
from typing import Any, Optional

from . import CommandContext, register

logger = logging.getLogger(__name__)


@register("request_diagnostics")
def request_diagnostics(
    command_id: int,
    payload: Optional[dict[str, Any]],
    ctx: CommandContext,
) -> None:
    """Payload: null. Gathers a snapshot of local service state and host info."""

    # 1. Acknowledge.
    ctx.ops.acknowledge_command(command_id)
    ctx.state.mark_executing(command_id)
    logger.info("request_diagnostics %s acknowledged", command_id)

    # 2. Gather diagnostics.
    diagnostics: dict[str, Any] = {
        "health": _safe(lambda: ctx.local.health()),
        "state": _safe(lambda: ctx.local.state()),
        "latest_measurement": _safe(lambda: ctx.local.latest_measurement()),
        "disk": _disk_info(),
        "agent": {
            "unit_id": ctx.agent_config.agent.unit_id,
            "version": ctx.agent_config.agent.version,
        },
    }

    # 3. POST /diagnostics — auto-completes the command server-side.
    ctx.ops.post_diagnostics(command_id, diagnostics)

    # Mirror the completion locally.
    ctx.state.mark_completed(command_id, result={"diagnostics_keys": list(diagnostics.keys())})
    logger.info("request_diagnostics %s completed", command_id)


def _safe(fn: Any) -> Any:
    """Diagnostics must not fail wholesale just because one sub-call errored."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _disk_info() -> dict[str, Any]:
    try:
        d = shutil.disk_usage("/")
        return {
            "total_bytes": d.total,
            "used_bytes": d.used,
            "free_bytes": d.free,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
