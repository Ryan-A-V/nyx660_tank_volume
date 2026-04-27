"""
FastAPI application — updated for continuous measurement loop.

New endpoints:
    GET  /measurements/latest     — latest measurement (non-blocking)
    GET  /measurements/history    — 24h measurement history
    GET  /loop/stats              — measurement loop performance
    GET  /store/stats             — SQLite storage stats

Existing endpoints preserved with same contracts.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import Response

from nyx660_tank_volume.api.models import CalibrateRequest, MeasureResponse
from nyx660_tank_volume.config import AppConfig
from nyx660_tank_volume.core.service import TankVolumeService
from nyx660_tank_volume.utils.images import depth_to_png_bytes


def build_app(cfg: AppConfig, service: TankVolumeService) -> FastAPI:
    app = FastAPI(title="Helios2 Tank Volume API", version="0.2.0")

    def require_token(
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> None:
        if x_api_key != cfg.server.api_token:
            raise HTTPException(status_code=401, detail="Invalid API token")

    # ------------------------------------------------------------------
    # Health & state
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        loop_stats = service.get_loop_stats()
        return {
            "status": "ok",
            "backend": cfg.camera.backend,
            "has_calibration": service.calibration is not None,
            "measurement_loop_running": loop_stats.get("is_running", False),
            "measurement_fps": loop_stats.get("fps", 0.0),
        }

    @app.get("/state", dependencies=[Depends(require_token)])
    def state() -> dict:
        latest = service.last_measurement
        frame = service.last_frame
        return {
            "has_calibration": service.calibration is not None,
            "last_measurement": latest.to_dict() if latest else None,
            "last_frame_timestamp_utc": frame.timestamp_utc
            if frame
            else None,
            "loop_stats": service.get_loop_stats(),
            "store_stats": service.get_store_stats(),
        }

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @app.post("/calibrate", dependencies=[Depends(require_token)])
    def calibrate(body: CalibrateRequest) -> dict:
        return service.calibrate_empty_tank(frames=body.frames)

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    @app.post(
        "/measure",
        response_model=MeasureResponse,
        dependencies=[Depends(require_token)],
    )
    def measure() -> MeasureResponse:
        """
        Return the latest measurement from the continuous loop.
        This no longer triggers a new acquisition — the loop runs
        independently.
        """
        result = service.measure()
        return MeasureResponse(**result.to_dict())

    @app.get("/measurements/latest", dependencies=[Depends(require_token)])
    def measurements_latest() -> dict:
        """Latest measurement as raw JSON (no Pydantic validation)."""
        latest = service.last_measurement
        if latest is None:
            raise HTTPException(
                status_code=503,
                detail="No measurement available yet",
            )
        return latest.to_dict()

    @app.get("/measurements/history", dependencies=[Depends(require_token)])
    def measurements_history(
        hours: Optional[float] = Query(
            default=24, description="Hours of history to return"
        ),
        since: Optional[str] = Query(
            default=None, description="ISO timestamp start"
        ),
        until: Optional[str] = Query(
            default=None, description="ISO timestamp end"
        ),
        limit: int = Query(
            default=1000, description="Max records to return"
        ),
    ) -> dict:
        """Return measurement history from local SQLite store."""
        records = service.get_measurement_history(
            hours=hours, since=since, until=until, limit=limit
        )
        return {
            "count": len(records),
            "measurements": records,
        }

    # ------------------------------------------------------------------
    # Depth preview
    # ------------------------------------------------------------------

    @app.get("/frame/depth.png", dependencies=[Depends(require_token)])
    def current_depth_png() -> Response:
        frame = service.capture_frame()
        png = depth_to_png_bytes(frame.depth_m)
        return Response(content=png, media_type="image/png")

    # ------------------------------------------------------------------
    # Loop and store stats
    # ------------------------------------------------------------------

    @app.get("/loop/stats", dependencies=[Depends(require_token)])
    def loop_stats() -> dict:
        """Measurement loop performance statistics."""
        return service.get_loop_stats()

    @app.get("/store/stats", dependencies=[Depends(require_token)])
    def store_stats() -> dict:
        """Local measurement storage statistics."""
        return service.get_store_stats()

    # ------------------------------------------------------------------
    # Testing mode controls
    # ------------------------------------------------------------------

    @app.post("/loop/trigger", dependencies=[Depends(require_token)])
    def trigger_measurement() -> dict:
        """
        Trigger the next measurement (testing mode only).
        In testing mode, the loop pauses after each measurement
        and waits for this endpoint to be called before proceeding.
        """
        accepted = service.trigger_next_measurement()
        if not accepted:
            raise HTTPException(
                status_code=409,
                detail="Trigger not accepted. Either testing mode is off "
                "or the loop is not waiting for a trigger.",
            )
        # Wait for the result
        result = service.wait_for_measurement_result(timeout=30.0)
        if result is None:
            raise HTTPException(
                status_code=504,
                detail="Measurement timed out after trigger.",
            )
        return result.to_dict()

    @app.post("/loop/testing-mode", dependencies=[Depends(require_token)])
    def set_testing_mode(
        enabled: bool = Query(
            ..., description="Enable or disable testing mode"
        ),
    ) -> dict:
        """Toggle testing mode on or off at runtime."""
        service.set_testing_mode(enabled)
        return {
            "testing_mode": enabled,
            "status": "enabled" if enabled else "disabled",
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @app.on_event("shutdown")
    def shutdown() -> None:
        service.stop()

    return app
