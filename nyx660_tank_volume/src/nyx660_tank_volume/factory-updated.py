from __future__ import annotations

from nyx660_tank_volume.config import CameraConfig

from .base import CameraBackend
from .mock_camera import MockNyx660Camera
from .mock_helios2_camera import MockHelios2WideCamera
from .scepter_camera import ScepterNyx660Camera
from .helios2_camera import Helios2WideCamera


def create_camera(cfg: CameraConfig) -> CameraBackend:
    backend = cfg.backend.lower().strip()

    # --- Original NYX660 backends ---
    if backend == "mock":
        return MockNyx660Camera(cfg.width, cfg.height, cfg.mount_height_m)

    if backend == "scepter":
        return ScepterNyx660Camera(
            width=cfg.width,
            height=cfg.height,
            device_index=cfg.device_index,
            depth_scale_m=cfg.depth_scale_m,
        )

    # --- Helios2 Wide backends ---
    if backend == "mock_helios2":
        return MockHelios2WideCamera(
            width=cfg.width,
            height=cfg.height,
            mount_height_m=cfg.mount_height_m,
            tank_depth_m=cfg.helios2.tank_depth_m if cfg.helios2 else 1.22,
            operating_mode=cfg.helios2.operating_mode if cfg.helios2 else "5000mm",
        )

    if backend == "helios2":
        if cfg.helios2 is None:
            raise ValueError(
                "camera.helios2 config section is required when using "
                "backend='helios2'. See config.example.yaml."
            )
        return Helios2WideCamera(
            width=cfg.width,
            height=cfg.height,
            device_index=cfg.device_index,
            operating_mode=cfg.helios2.operating_mode,
            exposure=cfg.helios2.exposure,
            spatial_filter=cfg.helios2.spatial_filter,
            confidence_threshold=cfg.helios2.confidence_threshold,
            image_accumulation=cfg.helios2.image_accumulation,
            conversion_gain=cfg.helios2.conversion_gain,
        )

    raise ValueError(f"Unsupported camera backend: {cfg.backend}")
