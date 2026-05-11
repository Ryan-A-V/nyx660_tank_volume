"""HTTP client to the local measurement service on 127.0.0.1:8080.

Wraps the existing FastAPI endpoints so commands don't import service internals.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class LocalApiError(RuntimeError):
    """Raised when the local measurement service returns an error or is unreachable."""


class LocalApiClient:
    def __init__(self, base_url: str, api_token: str, timeout_s: float = 15.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"x-api-key": api_token},
            timeout=httpx.Timeout(timeout_s),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LocalApiClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- low-level ---------------------------------------------------------

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        try:
            r = self._client.get(path, params=params)
        except httpx.HTTPError as e:
            raise LocalApiError(f"GET {path} failed: {e}") from e
        if r.status_code >= 400:
            raise LocalApiError(f"GET {path} returned {r.status_code}: {r.text}")
        return r.json() if r.content else None

    def _post(self, path: str, json_body: Optional[dict[str, Any]] = None) -> Any:
        try:
            r = self._client.post(path, json=json_body or {})
        except httpx.HTTPError as e:
            raise LocalApiError(f"POST {path} failed: {e}") from e
        if r.status_code >= 400:
            raise LocalApiError(f"POST {path} returned {r.status_code}: {r.text}")
        return r.json() if r.content else None

    # ---- typed wrappers ---------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def state(self) -> dict[str, Any]:
        return self._get("/state")

    def latest_measurement(self) -> dict[str, Any]:
        return self._get("/measurements/latest")

    def measurement_history(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        return self._get("/measurements/history", params=params)

    def trigger_calibration(self) -> dict[str, Any]:
        return self._post("/calibrate")

    def set_testing_mode(self, enabled: bool) -> dict[str, Any]:
        return self._post("/loop/testing-mode", {"enabled": enabled})

    def trigger_loop(self) -> dict[str, Any]:
        return self._post("/loop/trigger")

    def depth_png(self) -> bytes:
        try:
            r = self._client.get("/frame/depth.png")
        except httpx.HTTPError as e:
            raise LocalApiError(f"GET /frame/depth.png failed: {e}") from e
        if r.status_code >= 400:
            raise LocalApiError(
                f"GET /frame/depth.png returned {r.status_code}: {r.text}"
            )
        return r.content
