from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response

from nyx660_tank_volume.api.models import CalibrateRequest, MeasureResponse
from nyx660_tank_volume.config import AppConfig
from nyx660_tank_volume.core.service import TankVolumeService
from nyx660_tank_volume.utils.images import depth_to_png_bytes


def build_app(cfg: AppConfig, service: TankVolumeService) -> FastAPI:
    app = FastAPI(title="NYX660 Tank Volume API", version="0.1.0")

    def require_token(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key != cfg.server.api_token:
            raise HTTPException(status_code=401, detail="Invalid API token")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": cfg.camera.backend, "has_calibration": service.calibration is not None}

    @app.post("/calibrate", dependencies=[Depends(require_token)])
    def calibrate(body: CalibrateRequest) -> dict:
        return service.calibrate_empty_tank(frames=body.frames)

    @app.post("/measure", response_model=MeasureResponse, dependencies=[Depends(require_token)])
    def measure() -> MeasureResponse:
        result = service.measure()
        return MeasureResponse(**result.to_dict())

    @app.get("/frame/depth.png", dependencies=[Depends(require_token)])
    def current_depth_png() -> Response:
        frame = service.capture_frame()
        png = depth_to_png_bytes(frame.depth_m)
        return Response(content=png, media_type="image/png")

    @app.get("/state", dependencies=[Depends(require_token)])
    def state() -> dict:
        return {
            "has_calibration": service.calibration is not None,
            "last_measurement": None if service.last_measurement is None else service.last_measurement.to_dict(),
            "last_frame_timestamp_utc": None if service.last_frame is None else service.last_frame.timestamp_utc,
        }

    return app
