"""Main agent loop: build poll payload, POST /poll, handle response."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .alerts import compute_active_alerts
from .commands import CommandContext, get_handler
from .config import AgentConfig
from .config_hash import compute_config_hash, compute_dict_hash
from .local_api import LocalApiClient
from .ops_client import OpsClient, OpsCommandStateError, OpsError, OpsUnreachable
from .poll_builder import build_poll_payload
from .state import AgentStateStore, STATUS_EXECUTING

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        config: AgentConfig,
        ops: OpsClient,
        local: LocalApiClient,
        state: AgentStateStore,
    ) -> None:
        self._config = config
        self._ops = ops
        self._local = local
        self._state = state
        self._stop_event = threading.Event()
        self._command_ctx = CommandContext(
            ops=ops, local=local, state=state, agent_config=config
        )
        # Cached operator thresholds from the most recent OPS config payload.
        # Loaded from state.db on startup so we don't lose them across restarts.
        self._operator_thresholds: dict[str, Any] = state.get_ops_config_cache() or {}

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info(
            "agent loop starting (unit_id=%s, poll_interval=%.1fs)",
            self._config.agent.unit_id,
            self._config.poll.interval_s,
        )

        # On startup, reconcile any commands left in "executing" — agent crashed
        # mid-command. Mark them failed locally and tell OPS.
        self._reconcile_orphaned_executing()

        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                self._run_cycle()
            except Exception as e:  # noqa: BLE001
                # The loop must never die. Log and keep going.
                logger.exception("agent cycle raised: %s", e)

            elapsed = time.monotonic() - cycle_start
            remaining = max(0.0, self._config.poll.interval_s - elapsed)
            if remaining > 0:
                self._stop_event.wait(remaining)

        logger.info("agent loop stopped")

    # ---- crash recovery ---------------------------------------------------

    def _reconcile_orphaned_executing(self) -> None:
        """Commands left in `executing` after an agent restart never finished.
        Fail them locally and on OPS so operators see the truth."""
        for cmd in self._state.get_executing_commands():
            cmd_id = cmd["command_id"]
            error = "agent restarted mid-command"
            logger.warning("reconciling orphaned executing command %s: %s", cmd_id, error)
            try:
                self._ops.fail_command(cmd_id, error)
            except OpsError as e:
                # Best-effort — we still mark it failed locally so we don't loop.
                logger.warning("could not report failure of %s to OPS: %s", cmd_id, e)
            self._state.mark_failed(cmd_id, error)

    # ---- single cycle -----------------------------------------------------

    def _run_cycle(self) -> None:
        # 1. Compute current config hash from disk.
        config_hash = self._current_config_hash()

        # 2. Build poll payload (pulls /state from local service, builds alerts).
        try:
            state_snapshot = self._local.state()
        except Exception as e:  # noqa: BLE001
            logger.warning("local /state failed: %s", e)
            state_snapshot = {}

        active_alerts = compute_active_alerts(
            state_snapshot, self._config, self._operator_thresholds
        )

        poll_payload, _ = build_poll_payload(
            local=self._local,
            agent_config=self._config,
            config_hash=config_hash,
            active_alerts=active_alerts,
            operator_thresholds=self._operator_thresholds,
        )

        # 3. POST /poll.
        try:
            response = self._ops.poll(poll_payload)
        except OpsUnreachable as e:
            logger.warning("poll failed: %s", e)
            return
        except OpsError as e:
            logger.error("poll returned non-retryable error: %s", e)
            return

        # 4. Handle config delta.
        new_config = response.get("config")
        server_hash = response.get("config_hash")
        if new_config is not None:
            self._handle_config_delta(new_config, server_hash)

        # 5. Process commands.
        commands = response.get("commands") or []
        for cmd in commands:
            if self._stop_event.is_set():
                break
            self._process_command(cmd)

    # ---- config delta -----------------------------------------------------

    def _current_config_hash(self) -> Optional[str]:
        """Compute hash of the measurement service's current config.yaml.
        Returns None if the file can't be read (logged)."""
        try:
            return compute_config_hash(self._config.config_push.target_config_path)
        except FileNotFoundError:
            logger.warning(
                "config file not found at %s",
                self._config.config_push.target_config_path,
            )
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("could not hash config: %s", e)
            return None

    def _handle_config_delta(
        self, new_config: dict[str, Any], server_hash: Optional[str]
    ) -> None:
        """OPS sent us a new config. The push_config command (when it lands)
        will own the atomic-restart-with-rollback flow. For now we just cache
        the operator thresholds so alert evaluation uses them on the next cycle.

        TODO (next iteration): wire push_config / request_config commands to
        actually write the file, restart the service, and roll back on failure.
        """
        # Cache the operator thresholds used by alerts.
        # The OPS config is opaque to the agent — we just look for known keys.
        thresholds: dict[str, Any] = {}
        for key in (
            "tank_full_threshold",
            "tank_low_threshold",
            "min_valid_pixel_ratio",
            "calibration_max_age_seconds",
        ):
            if key in new_config:
                thresholds[key] = new_config[key]
        self._operator_thresholds = thresholds
        self._state.set_ops_config_cache(thresholds)

        # Sanity check: confirm we'd compute the same hash OPS computed.
        local_hash = compute_dict_hash(new_config)
        if server_hash and local_hash != server_hash:
            logger.warning(
                "config hash mismatch: server=%s local=%s — canonicalisation "
                "may differ between sides",
                server_hash, local_hash,
            )

        logger.info(
            "received new config from OPS (server_hash=%s); operator thresholds cached. "
            "Atomic apply will land with push_config handler.",
            server_hash,
        )

    # ---- command dispatch -------------------------------------------------

    def _process_command(self, cmd: dict[str, Any]) -> None:
        cmd_id = cmd.get("id")
        cmd_type = cmd.get("command")
        cmd_payload = cmd.get("payload_json")

        if not isinstance(cmd_id, int) or not cmd_type:
            logger.warning("skipping malformed command: %r", cmd)
            return

        # Idempotency: if we've already recorded this command, skip.
        recorded = self._state.record_received(cmd_id, cmd_type, cmd_payload)
        if not recorded:
            logger.debug("command %s already recorded, skipping", cmd_id)
            return

        try:
            handler = get_handler(cmd_type)
        except KeyError:
            error = f"unknown command type: {cmd_type}"
            logger.warning(error)
            try:
                self._ops.fail_command(cmd_id, error)
            except OpsError as e:
                logger.warning("could not fail %s on OPS: %s", cmd_id, e)
            self._state.mark_failed(cmd_id, error)
            return

        # Execute with timeout. We can't safely kill the thread on timeout
        # (Python limitation) — we stop waiting and mark failed. The thread
        # may continue running; its eventual side effects are discarded.
        result_holder: dict[str, Any] = {}

        def _runner() -> None:
            try:
                handler(cmd_id, cmd_payload, self._command_ctx)
                result_holder["ok"] = True
            except OpsCommandStateError as e:
                # 409/410 from OPS — the command's state changed under us.
                result_holder["error"] = f"command state error: {e}"
                # Mark locally — don't try to report to OPS, it already knows.
                self._state.mark_failed(cmd_id, str(e))
            except Exception as e:  # noqa: BLE001
                result_holder["error"] = str(e)

        worker = threading.Thread(target=_runner, daemon=True, name=f"cmd-{cmd_id}")
        worker.start()
        worker.join(timeout=self._config.poll.command_timeout_s)

        if worker.is_alive():
            error = f"command exceeded timeout of {self._config.poll.command_timeout_s}s"
            logger.warning("command %s (%s) %s", cmd_id, cmd_type, error)
            try:
                self._ops.fail_command(cmd_id, error)
            except OpsError as e:
                logger.warning("could not fail %s on OPS: %s", cmd_id, e)
            self._state.mark_failed(cmd_id, error)
            return

        if "error" in result_holder:
            error = result_holder["error"]
            logger.warning("command %s (%s) failed: %s", cmd_id, cmd_type, error)
            # If the handler already marked it failed (state error path), skip.
            try:
                self._ops.fail_command(cmd_id, error)
            except OpsError as e:
                logger.warning("could not fail %s on OPS: %s", cmd_id, e)
            # mark_failed is idempotent enough — the schema lets us overwrite.
            self._state.mark_failed(cmd_id, error)
