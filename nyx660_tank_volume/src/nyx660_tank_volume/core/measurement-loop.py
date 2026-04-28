"""
Continuous measurement loop with optional testing mode.

Normal mode:
    Runs continuously — as soon as one measurement completes,
    the next one begins.

Testing mode (config: measurement_loop.testing_mode: true):
    Pauses after each measurement and waits for an explicit
    trigger before taking the next one. Useful for validating
    results against known states (e.g., placing objects on a desk,
    adding known volumes to a tank, etc.)

The latest result is always available via get_latest() for the
API, WITS output, and cloud agent to read without blocking.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from nyx660_tank_volume.camera.base import CameraBackend, DepthFrame
from nyx660_tank_volume.config import AppConfig
from nyx660_tank_volume.core.calibration import CalibrationBundle
from nyx660_tank_volume.core.measurement import (
    MeasurementResult,
    estimate_volume,
    preprocess_depth,
    smooth_depth,
)
from nyx660_tank_volume.core.measurement_store import MeasurementStore

logger = logging.getLogger(__name__)


class MeasurementLoop:
    """
    Background thread that continuously measures tank volume.

    In testing mode, the loop pauses after each measurement and
    waits for trigger_next() to be called before proceeding.

    Lifecycle:
        loop = MeasurementLoop(cfg, camera, calibration, store)
        loop.start()
        ...
        latest = loop.get_latest()     # thread-safe, non-blocking

        # Testing mode only:
        loop.trigger_next()            # allow one measurement
        loop.wait_for_result(timeout)  # block until it completes
        ...
        loop.stop()
    """

    def __init__(
        self,
        cfg: AppConfig,
        camera: CameraBackend,
        calibration: Optional[CalibrationBundle],
        store: MeasurementStore,
    ) -> None:
        self.cfg = cfg
        self.camera = camera
        self.store = store
        self._calibration = calibration
        self._calibration_lock = threading.Lock()

        self._latest: Optional[MeasurementResult] = None
        self._latest_frame: Optional[DepthFrame] = None
        self._latest_lock = threading.Lock()

        self._frame_buffer: deque[np.ndarray] = deque(
            maxlen=max(1, cfg.measurement.smooth_frames)
        )

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Testing mode
        self._testing_mode: bool = (
            cfg.measurement_loop.testing_mode
            if cfg.measurement_loop is not None
            else False
        )
        self._trigger_event = threading.Event()
        self._result_ready_event = threading.Event()

        # Stats
        self._loop_count: int = 0
        self._error_count: int = 0
        self._last_loop_duration: float = 0.0
        self._started_at: Optional[str] = None
        self._waiting_for_trigger: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the continuous measurement loop in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Measurement loop is already running")
            return

        self._stop_event.clear()
        self._trigger_event.clear()
        self._result_ready_event.clear()

        if self._testing_mode:
            logger.info("Measurement loop starting in TESTING MODE")
        else:
            logger.info("Measurement loop starting in continuous mode")

        self._thread = threading.Thread(
            target=self._run_loop,
            name="measurement-loop",
            daemon=True,
        )
        self._thread.start()
        self._started_at = datetime.now(timezone.utc).isoformat()

    def stop(self) -> None:
        """Signal the loop to stop and wait for it to finish."""
        self._stop_event.set()
        # Unblock the loop if it's waiting for a trigger
        self._trigger_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.warning("Measurement loop thread did not exit cleanly")
        self._thread = None
        logger.info("Measurement loop stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def testing_mode(self) -> bool:
        return self._testing_mode

    @property
    def waiting_for_trigger(self) -> bool:
        return self._waiting_for_trigger

    def get_latest(self) -> Optional[MeasurementResult]:
        """Thread-safe read of the most recent measurement."""
        with self._latest_lock:
            return self._latest

    def get_latest_frame(self) -> Optional[DepthFrame]:
        """Thread-safe read of the most recent depth frame."""
        with self._latest_lock:
            return self._latest_frame

    def update_calibration(self, calibration: CalibrationBundle) -> None:
        """
        Hot-swap the calibration without stopping the loop.
        Called after a recalibration.
        """
        with self._calibration_lock:
            self._calibration = calibration
        logger.info("Measurement loop calibration updated")

    # ------------------------------------------------------------------
    # Testing mode controls
    # ------------------------------------------------------------------

    def trigger_next(self) -> bool:
        """
        Allow the next measurement to proceed (testing mode only).

        Returns True if the trigger was accepted (loop was waiting),
        False if not in testing mode or loop wasn't waiting.
        """
        if not self._testing_mode:
            logger.warning("trigger_next() called but not in testing mode")
            return False

        if not self._waiting_for_trigger:
            logger.warning(
                "trigger_next() called but loop is not waiting for a trigger"
            )
            return False

        self._result_ready_event.clear()
        self._trigger_event.set()
        logger.info("Measurement triggered")
        return True

    def wait_for_result(self, timeout: float = 30.0) -> Optional[MeasurementResult]:
        """
        Block until the triggered measurement completes (testing mode only).

        Returns the measurement result, or None if timed out.
        """
        if not self._testing_mode:
            return self.get_latest()

        if self._result_ready_event.wait(timeout=timeout):
            return self.get_latest()
        return None

    def set_testing_mode(self, enabled: bool) -> None:
        """
        Switch testing mode on or off at runtime.
        If disabling, unblocks the loop so it resumes continuous operation.
        """
        was_testing = self._testing_mode
        self._testing_mode = enabled

        if was_testing and not enabled:
            # Unblock if the loop was waiting for a trigger
            self._trigger_event.set()
            logger.info("Testing mode disabled — resuming continuous measurement")
        elif not was_testing and enabled:
            logger.info("Testing mode enabled — loop will pause after next measurement")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return loop performance statistics."""
        return {
            "is_running": self.is_running,
            "testing_mode": self._testing_mode,
            "waiting_for_trigger": self._waiting_for_trigger,
            "started_at": self._started_at,
            "loop_count": self._loop_count,
            "error_count": self._error_count,
            "last_loop_duration_s": round(self._last_loop_duration, 3),
            "fps": round(1.0 / self._last_loop_duration, 2)
            if self._last_loop_duration > 0
            else 0.0,
            "frame_buffer_size": len(self._frame_buffer),
            "has_calibration": self._calibration is not None,
        }

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main loop body. Runs until stop_event is set."""
        logger.info("Measurement loop thread running")

        while not self._stop_event.is_set():
            # In testing mode, wait for a trigger before each measurement
            if self._testing_mode:
                self._waiting_for_trigger = True
                logger.debug("Waiting for trigger...")
                self._trigger_event.wait()
                self._trigger_event.clear()
                self._waiting_for_trigger = False

                # Check if we were unblocked by stop()
                if self._stop_event.is_set():
                    break

                # Flush stale frames from the stream buffer.
                # While the loop was paused waiting for a trigger, the
                # camera kept streaming into its buffer. Without flushing,
                # we'd process an old frame from before the trigger instead
                # of a fresh one reflecting the current scene.
                self._flush_stale_frames()

            t_start = time.monotonic()

            try:
                produced_result = self._single_cycle()
            except Exception:
                self._error_count += 1
                logger.exception(
                    "Measurement loop error (count=%d)", self._error_count
                )
                # Brief pause after error to avoid tight spin
                self._stop_event.wait(timeout=2.0)
                continue

            self._last_loop_duration = time.monotonic() - t_start
            self._loop_count += 1

            # In testing mode, signal that the result is ready
            if self._testing_mode and produced_result:
                self._result_ready_event.set()

            if self._loop_count % 100 == 0:
                logger.info(
                    "Measurement loop: %d cycles, %.1f FPS, %d errors",
                    self._loop_count,
                    1.0 / self._last_loop_duration
                    if self._last_loop_duration > 0
                    else 0,
                    self._error_count,
                )

        logger.info(
            "Measurement loop exiting after %d cycles (%d errors)",
            self._loop_count,
            self._error_count,
        )

    def _flush_stale_frames(self, count: int = 15) -> None:
        """
        Discard frames to ensure the next get_frame() returns
        a fresh capture reflecting the current scene.

        The GigE stream pipeline and the sensor's multi-frequency
        acquisition mode can hold a significant number of buffered
        frames. We flush generously and add a brief pause to let
        the sensor complete any in-progress acquisition cycle.
        """
        import time

        # Brief pause to let any in-progress multi-freq capture complete
        time.sleep(0.5)

        for i in range(count):
            try:
                self.camera.get_frame()
            except Exception:
                pass

        # Clear the smoothing buffer so stale depth data
        # doesn't pollute the next measurement
        self._frame_buffer.clear()
        logger.debug("Flushed %d stale frames and cleared buffer", count)

    def _single_cycle(self) -> bool:
        """
        One measurement cycle.
        Returns True if a volume result was produced, False if only
        a frame was captured (no calibration or buffer not full yet).
        """
        # Capture
        frame = self.camera.get_frame()
        depth = preprocess_depth(frame.depth_m, self.cfg)
        self._frame_buffer.append(depth)

        # Update latest frame (for depth preview endpoint)
        processed_frame = DepthFrame(
            depth_m=depth,
            rgb=frame.rgb,
            timestamp_utc=frame.timestamp_utc,
        )
        with self._latest_lock:
            self._latest_frame = processed_frame

        # Need calibration and a full buffer to compute volume
        with self._calibration_lock:
            calibration = self._calibration

        if calibration is None:
            return False

        if len(self._frame_buffer) < self._frame_buffer.maxlen:
            return False

        # Compute volume
        smoothed = smooth_depth(list(self._frame_buffer))
        result = estimate_volume(smoothed, calibration, self.cfg)

        # Store
        with self._latest_lock:
            self._latest = result
        self.store.save(result)
        return True
