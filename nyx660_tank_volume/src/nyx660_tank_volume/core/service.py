"""
Tank volume service — orchestrates camera, calibration, measurement loop,
and local storage.

This replaces the original on-demand service.py. The key change is that
measurements now run continuously in a background thread rather than
being triggered per-request. API endpoints read the latest result from
the loop instead of blocking on a new acquisition.
"""

from __future__ import annotations

import logging
from typing import Optional
from datetime import datetime, timezone

import numpy as np

from nyx660_tank_volume.camera.base import CameraBackend, DepthFrame
from nyx660_tank_volume.config import AppConfig
from nyx660_tank_volume.core.calibration import (
    CalibrationBundle,
    CalibrationStore,
    create_calibration,
)
from nyx660_tank_volume.core.measurement import (
    MeasurementResult,
    preprocess_depth,
)
from nyx660_tank_volume.core.measurement_loop import MeasurementLoop
from nyx660_tank_volume.core.measurement_store import MeasurementStore
from nyx660_tank_volume.utils.io import save_json

logger = logging.getLogger(__name__)


class TankVolumeService:
    def __init__(
        self,
        cfg: AppConfig,
        camera: CameraBackend,
        calibration_store: CalibrationStore,
    ) -> None:
        self.cfg = cfg
        self.camera = camera
        self.calibration_store = calibration_store
        self.calibration: Optional[CalibrationBundle] = None

        # Local measurement history
        db_path = cfg.storage.data_dir + "/measurements.db"
        self.measurement_store = MeasurementStore(
            db_path=db_path,
            retention_hours=24,
        )

        # Continuous measurement loop (created at start)
        self._loop: Optional[MeasurementLoop] = None

    def start(self) -> None:
        """Open the camera, warm up, load calibration, start the loop."""
        self.camera.open()

        # Warmup frames — let the sensor stabilise
        logger.info("Warming up camera (%d frames)", self.cfg.camera.warmup_frames)
        for _ in range(self.cfg.camera.warmup_frames):
            self.camera.get_frame()

        # Load existing calibration if available
        if self.calibration_store.exists():
            self.calibration = self.calibration_store.load()
            logger.info("Loaded existing calibration from disk")

        # Start continuous measurement loop
        self._loop = MeasurementLoop(
            cfg=self.cfg,
            camera=self.camera,
            calibration=self.calibration,
            store=self.measurement_store,
        )
        self._loop.start()

    def stop(self) -> None:
        """Stop the loop and close the camera."""
        if self._loop is not None:
            self._loop.stop()
        self.camera.close()

    # ------------------------------------------------------------------
    # Calibration (pauses the loop briefly)
    # ------------------------------------------------------------------

    def calibrate_empty_tank(self, frames: Optional[int] = None) -> dict:
        """
        Run calibration against an empty tank.

        The measurement loop continues running during calibration but
        won't produce valid volume results until the new calibration
        is applied (it just captures frames without computing volume
        if calibration is None or being replaced).
        """
        frame_count = int(frames or self.cfg.measurement.baseline_frames)
        logger.info("Starting calibration with %d frames", frame_count)

        # Capture calibration frames directly (not through the loop)
        samples: list[np.ndarray] = []
        for i in range(frame_count):
            frame = self.camera.get_frame()
            depth = preprocess_depth(frame.depth_m, self.cfg)
            samples.append(depth)

        # Build and save calibration
        self.calibration = create_calibration(samples, self.cfg)
        self.calibration_store.save(self.calibration)
        logger.info(
            "Calibration complete: footprint=%.2f m²",
            self.calibration.footprint_area_m2,
        )

        # Hot-swap calibration into the running loop
        if self._loop is not None:
            self._loop.update_calibration(self.calibration)

        response = {
            "status": "ok",
            "created_utc": self.calibration.created_utc,
            "footprint_area_m2": self.calibration.footprint_area_m2,
            "frame_count": frame_count,
            "detection": self.calibration.detection_info,
        }

        # Surface warnings to the top level for visibility
        warnings = self.calibration.detection_info.get("warnings", [])
        if warnings:
            response["warnings"] = warnings

        return response

    # ------------------------------------------------------------------
    # Read latest data (non-blocking, reads from the loop)
    # ------------------------------------------------------------------

    @property
    def last_measurement(self) -> Optional[MeasurementResult]:
        """Latest measurement from the continuous loop."""
        if self._loop is not None:
            return self._loop.get_latest()
        return None

    @property
    def last_frame(self) -> Optional[DepthFrame]:
        """Latest depth frame from the continuous loop."""
        if self._loop is not None:
            return self._loop.get_latest_frame()
        return None

    def capture_frame(self) -> DepthFrame:
        """
        Get the latest frame. Prefers the loop's cached frame.
        Falls back to a direct capture if the loop isn't running.
        """
        if self._loop is not None:
            frame = self._loop.get_latest_frame()
            if frame is not None:
                return frame
        # Fallback: direct capture
        frame = self.camera.get_frame()
        depth = preprocess_depth(frame.depth_m, self.cfg)
        return DepthFrame(
            depth_m=depth, rgb=frame.rgb, timestamp_utc=frame.timestamp_utc
        )

    def measure(self) -> MeasurementResult:
        """
        Get the latest measurement result.

        In the original code this triggered a new acquisition.
        Now it reads the latest result from the continuous loop.
        If no result is available yet, raises an error.
        """
        result = self.last_measurement
        if result is None:
            raise RuntimeError(
                "No measurement available yet. Ensure calibration has been "
                "run and the measurement loop has completed at least one cycle."
            )
        save_json(
            self.cfg.storage.latest_measurement_file, result.to_dict()
        )
        return result

    # ------------------------------------------------------------------
    # Loop and store access
    # ------------------------------------------------------------------

    def get_loop_stats(self) -> dict:
        """Return measurement loop performance stats."""
        if self._loop is not None:
            return self._loop.get_stats()
        return {"is_running": False}

    def get_measurement_history(
        self,
        hours: Optional[float] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 10000,
    ) -> list[dict]:
        """Return measurement history from local SQLite store."""
        return self.measurement_store.get_history(
            hours=hours, since=since, until=until, limit=limit
        )

    def get_store_stats(self) -> dict:
        """Return storage statistics."""
        return self.measurement_store.get_stats()

    # ------------------------------------------------------------------
    # Testing mode controls
    # ------------------------------------------------------------------

    def trigger_next_measurement(self) -> bool:
        """Trigger the next measurement in testing mode."""
        if self._loop is not None:
            return self._loop.trigger_next()
        return False

    def wait_for_measurement_result(
        self, timeout: float = 30.0
    ) -> Optional[MeasurementResult]:
        """Block until the triggered measurement completes."""
        if self._loop is not None:
            return self._loop.wait_for_result(timeout=timeout)
        return None

    def set_testing_mode(self, enabled: bool) -> None:
        """Toggle testing mode on or off at runtime."""
        if self._loop is not None:
            self._loop.set_testing_mode(enabled)
        logger.info("Testing mode %s", "enabled" if enabled else "disabled")


    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def is_sensor_connected(self) -> bool:
        """True when the camera backend has an active sensor connection."""
        return self.camera.is_open if self.camera else False

    def is_streaming(self) -> bool:
        """True when the loop is alive AND a fresh frame arrived recently.
        'Recently' is 2x the expected frame interval based on loop FPS."""
        if not self._loop or not self._loop.is_running():
            return False
        last_frame_ts = self._loop.last_frame_timestamp_utc()
        if last_frame_ts is None:
            return False
        age_seconds = (datetime.now(timezone.utc) - last_frame_ts).total_seconds()
        fps = self._loop.fps() or 1.0
        max_age = max(2.0 / fps, 1.0)   # at least 1 second tolerance
        return age_seconds <= max_age

    def calibration_age_seconds(self) -> Optional[float]:
        """Seconds since the current calibration was created.
        Returns None if no calibration is loaded."""
        if self.calibration is None:
            return None
        created_at = getattr(self.calibration, "created_at_utc", None)
        if created_at is None:
            return None
        return (datetime.now(timezone.utc) - created_at).total_seconds()