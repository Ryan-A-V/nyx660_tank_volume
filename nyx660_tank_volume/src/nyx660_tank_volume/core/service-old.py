from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from nyx660_tank_volume.camera.base import CameraBackend, DepthFrame
from nyx660_tank_volume.config import AppConfig
from nyx660_tank_volume.core.calibration import CalibrationStore, CalibrationBundle, create_calibration
from nyx660_tank_volume.core.measurement import MeasurementResult, estimate_volume, preprocess_depth, smooth_depth
from nyx660_tank_volume.utils.io import save_json


class TankVolumeService:
    def __init__(self, cfg: AppConfig, camera: CameraBackend, calibration_store: CalibrationStore) -> None:
        self.cfg = cfg
        self.camera = camera
        self.calibration_store = calibration_store
        self.calibration: Optional[CalibrationBundle] = None
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=max(1, cfg.measurement.smooth_frames))
        self.last_frame: Optional[DepthFrame] = None
        self.last_measurement: Optional[MeasurementResult] = None

    def start(self) -> None:
        self.camera.open()
        for _ in range(self.cfg.camera.warmup_frames):
            self.capture_frame()
        if self.calibration_store.exists():
            self.calibration = self.calibration_store.load()

    def stop(self) -> None:
        self.camera.close()

    def capture_frame(self) -> DepthFrame:
        frame = self.camera.get_frame()
        depth = preprocess_depth(frame.depth_m, self.cfg)
        self.frame_buffer.append(depth)
        self.last_frame = DepthFrame(depth_m=depth, rgb=frame.rgb, timestamp_utc=frame.timestamp_utc)
        return self.last_frame

    def capture_smoothed_depth(self) -> np.ndarray:
        while len(self.frame_buffer) < self.frame_buffer.maxlen:
            self.capture_frame()
        return smooth_depth(list(self.frame_buffer))

    def calibrate_empty_tank(self, frames: Optional[int] = None) -> dict:
        frame_count = int(frames or self.cfg.measurement.baseline_frames)
        self.frame_buffer.clear()
        samples: list[np.ndarray] = []
        for _ in range(frame_count):
            frame = self.capture_frame()
            samples.append(frame.depth_m)
        self.calibration = create_calibration(samples, self.cfg)
        self.calibration_store.save(self.calibration)
        return {
            "status": "ok",
            "created_utc": self.calibration.created_utc,
            "footprint_area_m2": self.calibration.footprint_area_m2,
            "frame_count": frame_count,
        }

    def measure(self) -> MeasurementResult:
        if self.calibration is None:
            raise RuntimeError("No calibration found. Run /calibrate first with an empty tank.")
        self.frame_buffer.clear()
        current = self.capture_smoothed_depth()
        result = estimate_volume(current, self.calibration, self.cfg)
        self.last_measurement = result
        save_json(self.cfg.storage.latest_measurement_file, result.to_dict())
        return result
