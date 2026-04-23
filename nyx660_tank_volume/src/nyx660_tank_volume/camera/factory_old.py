from __future__ import annotations

from nyx660_tank_volume.config import CameraConfig

from .base import CameraBackend
from .mock_camera import MockNyx660Camera
from .scepter_camera import ScepterNyx660Camera


def create_camera(cfg: CameraConfig) -> CameraBackend:
    backend = cfg.backend.lower().strip()
    if backend == "mock":
        return MockNyx660Camera(cfg.width, cfg.height, cfg.mount_height_m)
    if backend == "scepter":
        return ScepterNyx660Camera(
            width=cfg.width,
            height=cfg.height,
            device_index=cfg.device_index,
            depth_scale_m=cfg.depth_scale_m,
        )
    raise ValueError(f"Unsupported camera backend: {cfg.backend}")
