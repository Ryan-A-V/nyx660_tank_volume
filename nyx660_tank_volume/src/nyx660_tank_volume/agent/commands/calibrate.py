"""calibrate — run a calibration capture and complete via /commands/{id}/complete.

This is the pattern for commands that don't have a dedicated upload endpoint:
ack, do the work, then explicitly call /complete (or /fail) with the result.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from . import CommandContext, register

logger = logging.getLogger(__name__)


@register("calibrate")
def calibrate(
    command_id: int,
    payload: Optional[dict[str, Any]],
    ctx: CommandContext,
) -> None:
    """Payload: {"frame_count": int} (optional — local service uses its
    configured default if not provided)."""

    # 1. Acknowledge — pending → executing.
    ctx.ops.acknowledge_command(command_id)
    ctx.state.mark_executing(command_id)
    logger.info("calibrate %s acknowledged", command_id)

    started_at = time.monotonic()

    # 2. Do the work.
    # The local service's /calibrate uses whatever frame count is in its
    # config.yaml. If OPS provides a different frame_count we'd need a way
    # to override it on the local side — for now we just pass through any
    # received hint to OPS in the result for traceability.
    local_result = ctx.local.trigger_calibration()
    duration_ms = int((time.monotonic() - started_at) * 1000)

    # 3. Complete with a result.
    result: dict[str, Any] = {
        "duration_ms": duration_ms,
        "local_result": local_result,
    }
    if payload and "frame_count" in payload:
        result["requested_frame_count"] = payload["frame_count"]

    ctx.ops.complete_command(command_id, result=result)
    ctx.state.mark_completed(command_id, result=result)
    logger.info("calibrate %s completed in %dms", command_id, duration_ms)
