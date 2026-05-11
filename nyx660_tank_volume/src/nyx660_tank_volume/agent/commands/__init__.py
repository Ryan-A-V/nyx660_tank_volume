"""Command registry and base types.

Each handler is responsible for the FULL command lifecycle on its own:

  1. Call ctx.ops.acknowledge_command(command_id) — moves pending → executing.
     Update ctx.state with mark_executing(command_id).
  2. Do the work.
  3. Either:
       a) Call a dedicated upload endpoint (post_diagnostics, post_depth_preview,
          post_measurement_history) which auto-completes server-side, OR
       b) Call ctx.ops.complete_command(command_id, result).
     Update ctx.state with mark_completed(command_id, result).
  4. On any exception inside the handler, the loop catches it and calls
     ctx.ops.fail_command(command_id, error_message). Handlers can also call
     fail_command themselves for partial-success scenarios.

The reason for this design (vs the loop owning ack/complete): some commands
complete via auto-completing endpoints, and the handler is the only thing
that knows which path it takes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..local_api import LocalApiClient
from ..ops_client import OpsClient
from ..state import AgentStateStore
from ..config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class CommandContext:
    """Resources available to every command handler."""

    ops: OpsClient
    local: LocalApiClient
    state: AgentStateStore
    agent_config: AgentConfig


# Handler signature: takes the OPS-assigned command_id, the payload (or None
# for commands that have no payload), and the context. Returns nothing —
# handlers report success/failure to OPS themselves.
CommandHandler = Callable[[int, Optional[Dict[str, Any]], CommandContext], None]


_REGISTRY: Dict[str, CommandHandler] = {}


def register(command_type: str) -> Callable[[CommandHandler], CommandHandler]:
    def decorator(fn: CommandHandler) -> CommandHandler:
        if command_type in _REGISTRY:
            raise RuntimeError(f"Command already registered: {command_type}")
        _REGISTRY[command_type] = fn
        return fn

    return decorator


def get_handler(command_type: str) -> CommandHandler:
    if command_type not in _REGISTRY:
        raise KeyError(f"Unknown command type: {command_type}")
    return _REGISTRY[command_type]


def known_command_types() -> list[str]:
    return sorted(_REGISTRY.keys())


# ---- handler imports (registers them as a side effect) -------------------
# Keep this at the bottom to avoid circular imports. Only the two example
# handlers are imported here in the skeleton; remaining handlers get added
# in the next iteration.
from . import calibrate  # noqa: E402, F401
from . import request_diagnostics  # noqa: E402, F401
