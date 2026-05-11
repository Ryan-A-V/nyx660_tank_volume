"""HTTP client to the OPS lidar API."""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)


class OpsError(RuntimeError):
    """Base for OPS client errors."""


class OpsUnreachable(OpsError):
    """Raised when OPS can't be reached after retries (network, 5xx)."""


class OpsCommandStateError(OpsError):
    """Raised on 409 or 410 — command state changed under us. Terminal: don't
    retry, let the caller decide what to do."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpsClient:
    """Synchronous HTTP client for the OPS lidar API.

    Auth is a single bearer token in the Authorization header. OPS resolves
    the unit from the token, so the client never sends unit_id explicitly.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        request_timeout_s: float = 30.0,
        connect_timeout_s: float = 10.0,
        backoff_initial_s: float = 5.0,
        backoff_max_s: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._backoff_initial_s = backoff_initial_s
        self._backoff_max_s = backoff_max_s
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(
                request_timeout_s,
                connect=connect_timeout_s,
            ),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpsClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- internal: request with retry/backoff -----------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        max_attempts: Optional[int] = None,
        on_attempt: Optional[Callable[[int, Exception], None]] = None,
    ) -> dict[str, Any]:
        """Run an HTTP request with optional exponential backoff.

        Behavior:
          - 2xx → return parsed JSON (or empty dict if no body).
          - 401 → raise OpsError immediately (auth failure, no retry).
          - 409 / 410 → raise OpsCommandStateError immediately (terminal).
          - 422 → raise OpsError immediately (bad payload, no retry).
          - 4xx (other) → raise OpsError, no retry.
          - 5xx / network errors → retry with jittered exponential backoff.
          - max_attempts=None means retry indefinitely (capped delay).
          - max_attempts=1 means single attempt (used for /poll itself, since
            the loop owns the inter-poll backoff).
        """
        attempt = 0
        delay = self._backoff_initial_s
        while True:
            attempt += 1
            try:
                if files is not None:
                    r = self._client.request(
                        method, path, files=files, data=data or {}
                    )
                else:
                    r = self._client.request(method, path, json=json_body)
            except httpx.HTTPError as e:
                if on_attempt:
                    on_attempt(attempt, e)
                if max_attempts is not None and attempt >= max_attempts:
                    raise OpsUnreachable(
                        f"{method} {path} failed after {attempt} attempts: {e}"
                    ) from e
                self._sleep_backoff(attempt, delay, e, method, path)
                delay = min(delay * 2.0, self._backoff_max_s)
                continue

            # Successful HTTP exchange — examine the status code.
            if 200 <= r.status_code < 300:
                if not r.content:
                    return {}
                return r.json()

            if r.status_code == 401:
                raise OpsError(f"{method} {path} returned 401: {r.text}")
            if r.status_code in (409, 410):
                raise OpsCommandStateError(
                    r.status_code, f"{method} {path} returned {r.status_code}: {r.text}"
                )
            if r.status_code == 422:
                raise OpsError(f"{method} {path} returned 422 (validation): {r.text}")
            if 400 <= r.status_code < 500:
                # Other 4xx: not retryable.
                raise OpsError(
                    f"{method} {path} returned {r.status_code}: {r.text}"
                )

            # 5xx: retryable.
            err = OpsUnreachable(
                f"{method} {path} returned {r.status_code}: {r.text}"
            )
            if on_attempt:
                on_attempt(attempt, err)
            if max_attempts is not None and attempt >= max_attempts:
                raise err
            self._sleep_backoff(attempt, delay, err, method, path)
            delay = min(delay * 2.0, self._backoff_max_s)

    def _sleep_backoff(
        self,
        attempt: int,
        delay: float,
        err: Exception,
        method: str,
        path: str,
    ) -> None:
        sleep_for = min(delay, self._backoff_max_s)
        sleep_for *= 0.5 + random.random()  # 0.5x–1.5x jitter
        logger.warning(
            "OPS %s %s attempt %d failed (%s); sleeping %.1fs",
            method, path, attempt, err, sleep_for,
        )
        time.sleep(sleep_for)

    # ---- endpoints --------------------------------------------------------

    def poll(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /poll. Single attempt per call — the agent loop owns the
        inter-poll backoff cadence."""
        return self._request("POST", "/poll", json_body=payload, max_attempts=1)

    def get_config(self) -> dict[str, Any]:
        """GET /config — fetch the unit's currently active config without polling."""
        return self._request("GET", "/config", max_attempts=3)

    def post_measurement(self, measurement: dict[str, Any]) -> dict[str, Any]:
        """POST /measurements — out-of-band single measurement push."""
        return self._request("POST", "/measurements", json_body=measurement, max_attempts=3)

    def post_measurement_history(
        self, command_id: int, measurements: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """POST /measurement-history — bulk insert in response to request_history.
        Auto-completes the originating command."""
        return self._request(
            "POST",
            "/measurement-history",
            json_body={"command_id": command_id, "measurements": measurements},
            max_attempts=3,
        )

    def post_diagnostics(
        self, command_id: int, diagnostics: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /diagnostics — diagnostic dump in response to request_diagnostics.
        Auto-completes the originating command."""
        return self._request(
            "POST",
            "/diagnostics",
            json_body={"command_id": command_id, "diagnostics": diagnostics},
            max_attempts=3,
        )

    def post_depth_preview(
        self, command_id: int, png_bytes: bytes
    ) -> dict[str, Any]:
        """POST /depth-preview as multipart/form-data. Auto-completes the
        originating command."""
        return self._request(
            "POST",
            "/depth-preview",
            files={"image": ("depth.png", png_bytes, "image/png")},
            data={"command_id": str(command_id)},
            max_attempts=3,
        )

    # ---- command lifecycle ------------------------------------------------

    def acknowledge_command(self, command_id: int) -> dict[str, Any]:
        """POST /commands/{id}/acknowledge — move pending → executing."""
        return self._request(
            "POST",
            f"/commands/{command_id}/acknowledge",
            json_body={},
            max_attempts=3,
        )

    def complete_command(
        self, command_id: int, result: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """POST /commands/{id}/complete — mark executing → completed."""
        body = {"result": result} if result is not None else {}
        return self._request(
            "POST", f"/commands/{command_id}/complete", json_body=body, max_attempts=3
        )

    def fail_command(self, command_id: int, error: str) -> dict[str, Any]:
        """POST /commands/{id}/fail — mark pending/executing → failed."""
        return self._request(
            "POST",
            f"/commands/{command_id}/fail",
            json_body={"error": error},
            max_attempts=3,
        )
