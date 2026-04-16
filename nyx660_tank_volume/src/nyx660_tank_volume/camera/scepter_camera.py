from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .base import CameraBackend, DepthFrame


class ScepterNyx660Camera(CameraBackend):
    """
    Thin adapter for the Vzense/Scepter Python SDK.

    This project keeps the rest of the stack pure Python. The SDK binding is the
    only place expected to need vendor-specific changes because the exact module
    path and frame APIs can vary by SDK release.
    """

    def __init__(self, width: int, height: int, device_index: int = 0, depth_scale_m: float = 0.001) -> None:
        self.width = width
        self.height = height
        self.device_index = device_index
        self.depth_scale_m = depth_scale_m
        self.sdk: Optional[Any] = None
        self.device: Optional[Any] = None

    def open(self) -> None:
        try:
            # Replace this block with the actual ScepterSDK import path on your Jetson.
            # Example idea only:
            # from ScepterSDK import DeviceManager
            # self.sdk = DeviceManager()
            # devices = self.sdk.enumerate_devices()
            # self.device = devices[self.device_index]
            # self.device.open()
            raise NotImplementedError(
                "Hook the actual ScepterSDK Python import and frame retrieval here."
            )
        except Exception as exc:
            raise RuntimeError(
                "ScepterSDK backend is not wired yet. Use backend=mock to validate the stack, "
                "then patch scepter_camera.py on the Jetson with the vendor SDK calls."
            ) from exc

    def close(self) -> None:
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
        self.device = None
        self.sdk = None

    def get_frame(self) -> DepthFrame:
        if self.device is None:
            raise RuntimeError("Camera is not open")

        # Pseudocode for the vendor SDK:
        # frame = self.device.get_depth_frame()
        # depth_raw = np.frombuffer(frame.data, dtype=np.uint16).reshape(self.height, self.width)
        # depth_m = depth_raw.astype(np.float32) * self.depth_scale_m

        raise NotImplementedError(
            "Implement vendor frame acquisition here once the SDK is installed on the Jetson."
        )

        # return DepthFrame(
        #     depth_m=depth_m,
        #     timestamp_utc=datetime.now(timezone.utc).isoformat(),
        #     rgb=None,
        # )
