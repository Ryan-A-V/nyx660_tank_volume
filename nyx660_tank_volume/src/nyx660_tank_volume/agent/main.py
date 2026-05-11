"""nyx660-agent entrypoint."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from .config import load_agent_config
from .local_api import LocalApiClient
from .loop import AgentLoop
from .ops_client import OpsClient
from .state import AgentStateStore


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nyx660-agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("agent.yaml"),
        help="Path to agent.yaml (default: ./agent.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    log = logging.getLogger("nyx660_agent")

    try:
        config = load_agent_config(args.config)
    except Exception as e:  # noqa: BLE001
        log.error("failed to load agent config: %s", e)
        return 2

    state = AgentStateStore(config.storage.db_path)

    with (
        LocalApiClient(
            base_url=config.local.base_url,
            api_token=config.local.api_token,
            timeout_s=config.local.request_timeout_s,
        ) as local,
        OpsClient(
            base_url=config.ops.base_url,
            api_token=config.ops.api_token,  # type: ignore[arg-type]
            request_timeout_s=config.ops.request_timeout_s,
            connect_timeout_s=config.ops.connect_timeout_s,
            backoff_initial_s=config.ops.backoff_initial_s,
            backoff_max_s=config.ops.backoff_max_s,
        ) as ops,
    ):
        loop = AgentLoop(config=config, ops=ops, local=local, state=state)

        def _on_signal(signum: int, _frame) -> None:
            log.info("received signal %s, shutting down", signum)
            loop.stop()

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

        loop.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
