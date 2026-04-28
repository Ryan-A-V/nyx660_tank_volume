from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class CropConfig(BaseModel):
    enabled: bool = False
    x_min: int = 0
    x_max: int = 0
    y_min: int = 0
    y_max: int = 0


class IntrinsicsConfig(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class Helios2Config(BaseModel):
    """Helios2 Wide specific settings. Only used when backend is 'helios2' or 'mock_helios2'."""

    operating_mode: str = "5000mm"
    exposure: str = "long"
    spatial_filter: bool = True
    confidence_threshold: bool = True
    image_accumulation: int = 1
    conversion_gain: str = "Low"
    tank_depth_m: float = 1.22


class CameraConfig(BaseModel):
    backend: str = "mock"
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 5
    depth_scale_m: float = 0.001
    warmup_frames: int = 20
    mount_height_m: float = 2.5
    crop: CropConfig = Field(default_factory=CropConfig)
    intrinsics: IntrinsicsConfig
    helios2: Optional[Helios2Config] = None


class MeasurementConfig(BaseModel):
    min_valid_depth_m: float = 0.2
    max_valid_depth_m: float = 8.5
    smooth_frames: int = 5
    baseline_frames: int = 30
    fill_threshold_m: float = 0.015
    max_height_step_m: float = 2.0
    known_volume_liters: Optional[float] = None
    outlier_clip_percentile_low: float = 2.0
    outlier_clip_percentile_high: float = 98.0
    occupancy_mask_min_height_m: float = 0.01


class StorageConfig(BaseModel):
    data_dir: str = "./data"
    calibration_file: str = "./data/empty_tank_calibration.npz"
    latest_measurement_file: str = "./data/latest_measurement.json"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    api_token: str = "change-me"


class MeasurementLoopConfig(BaseModel):
    """Controls the continuous measurement loop behaviour."""

    testing_mode: bool = True


class AppConfig(BaseModel):
    server: ServerConfig
    storage: StorageConfig
    camera: CameraConfig
    measurement: MeasurementConfig
    measurement_loop: Optional[MeasurementLoopConfig] = Field(
        default_factory=MeasurementLoopConfig
    )


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = AppConfig.model_validate(raw)
    Path(cfg.storage.data_dir).mkdir(parents=True, exist_ok=True)
    return cfg
