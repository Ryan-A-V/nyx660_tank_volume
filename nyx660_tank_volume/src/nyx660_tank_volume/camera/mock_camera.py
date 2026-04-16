from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .base import CameraBackend, DepthFrame


class MockNyx660Camera(CameraBackend):
    def __init__(self, width: int, height: int, mount_height_m: float) -> None:
        self.width = width
        self.height = height
        self.mount_height_m = mount_height_m
        self.rng = np.random.default_rng(42)
        self.opened = False
        self.mode = "filled"

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def get_frame(self) -> DepthFrame:
        if not self.opened:
            raise RuntimeError("Camera is not open")
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        base = np.full((self.height, self.width), self.mount_height_m, dtype=np.float32)

        if self.mode == "empty":
            depth = base
        else:
            cx = self.width * 0.5
            cy = self.height * 0.5
            rx = self.width * 0.28
            ry = self.height * 0.22
            mound = 0.45 * np.exp(-(((xx - cx) ** 2) / (2 * rx**2) + ((yy - cy) ** 2) / (2 * ry**2)))
            ridge = 0.18 * np.exp(-((yy - self.height * 0.35) ** 2) / (2 * (self.height * 0.08) ** 2))
            height = np.clip(mound + ridge, 0.0, 0.8)
            depth = base - height

        noise = self.rng.normal(0.0, 0.004, size=depth.shape).astype(np.float32)
        depth = np.clip(depth + noise, 0.15, None)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb[..., 1] = 80
        return DepthFrame(
            depth_m=depth.astype(np.float32),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            rgb=rgb,
        )
