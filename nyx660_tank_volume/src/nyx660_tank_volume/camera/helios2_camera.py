"""
Helios2 Wide camera backend using LUCID Arena SDK.

Implements CameraBackend using Coord3D_C16 (depth-only) pixel format
for efficient depth acquisition. Returns depth in metres as a float32
NumPy array matching the DepthFrame contract.

Requirements beyond the base project:
    - arena_api wheel installed (from LUCID Arena SDK download)
    - ArenaC native library on the system library path

Everything else (numpy, etc.) is already in requirements.txt.
"""

from __future__ import annotations

import ctypes
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .base import CameraBackend, DepthFrame

logger = logging.getLogger(__name__)

# Distance mode lookup — maps config strings to Arena node values.
# The Helios2 Wide supports these operating modes. Pick the one that
# covers your sensor-to-floor distance with the best accuracy/precision.
OPERATING_MODES = {
    "1250mm": "Distance1250mmSingleFreq",
    "3000mm": "Distance3000mmSingleFreq",
    "4000mm": "Distance4000mmSingleFreq",
    "5000mm": "Distance5000mmMultiFreq",
    "6000mm": "Distance6000mmSingleFreq",
    "8300mm": "Distance8300mmMultiFreq",
}

# Exposure time selector lookup
EXPOSURE_SELECTORS = {
    "short": "Exp187_5Us",
    "medium": "Exp750Us",
    "long": "Exp3000Us",
}

# Note: The Helios2 Wide (HTW003S) only supports these three exposure
# values. The standard Helios2 (HLT/HTP) uses Exp62_5Us, Exp250Us,
# and Exp1000Us instead. This mapping is correct for the Wide model.


class Helios2WideCamera(CameraBackend):
    """
    Adapter for the LUCID Helios2 Wide (HTW003S) via arena_api.

    Streams Coord3D_C16 frames (16-bit unsigned depth per pixel),
    converts to metres using the camera's Scan3d coordinate scale
    and offset, and returns a standard DepthFrame.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        device_index: int = 0,
        operating_mode: str = "5000mm",
        exposure: str = "long",
        spatial_filter: bool = True,
        confidence_threshold: bool = True,
        image_accumulation: int = 1,
        conversion_gain: str = "Low",
        stream_packet_negotiate: bool = True,
        stream_packet_resend: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.device_index = device_index
        self.operating_mode = operating_mode
        self.exposure = exposure
        self.spatial_filter = spatial_filter
        self.confidence_threshold = confidence_threshold
        self.image_accumulation = image_accumulation
        self.conversion_gain = conversion_gain
        self.stream_packet_negotiate = stream_packet_negotiate
        self.stream_packet_resend = stream_packet_resend

        # Set at open() time
        self._system: Any = None
        self._device: Any = None
        self._scale_z: float = 1.0
        self._offset_z: float = 0.0
        self._streaming: bool = False

    # ------------------------------------------------------------------
    # CameraBackend interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        try:
            from arena_api.system import system
        except ImportError as exc:
            raise RuntimeError(
                "arena_api is not installed. Install the Arena SDK Python "
                "wheel from LUCID's download page before using the helios2 "
                "backend. Use backend=mock_helios2 to validate the stack "
                "without the SDK."
            ) from exc

        self._system = system

        # Discover devices
        devices = self._system.create_device()
        if not devices:
            raise RuntimeError(
                "No LUCID camera found on the network. Check Ethernet "
                "connection, IP configuration, and jumbo frame settings."
            )
        if self.device_index >= len(devices):
            raise RuntimeError(
                f"Device index {self.device_index} requested but only "
                f"{len(devices)} device(s) found."
            )

        self._device = devices[self.device_index]
        model = self._device.nodemap["DeviceModelName"].value
        serial = self._device.nodemap["DeviceSerialNumber"].value
        logger.info("Connected to %s (serial %s)", model, serial)

        self._configure_device()
        self._read_coordinate_params()
        self._start_stream()

    def close(self) -> None:
        if self._streaming and self._device is not None:
            try:
                self._device.stop_stream()
            except Exception:
                pass
            self._streaming = False
        if self._system is not None:
            try:
                self._system.destroy_device()
            except Exception:
                pass
        self._device = None
        self._system = None

    def get_frame(self) -> DepthFrame:
        if self._device is None or not self._streaming:
            raise RuntimeError("Camera is not open or not streaming")

        buffer = self._device.get_buffer()
        try:
            depth_m = self._buffer_to_depth_m(buffer)
        finally:
            self._device.requeue_buffer(buffer)

        return DepthFrame(
            depth_m=depth_m,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            rgb=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_device(self) -> None:
        """Apply all Helios2-specific node settings before streaming."""
        nodemap = self._device.nodemap

        # Pixel format — Coord3D_C16 gives depth-only, 16 bits unsigned
        nodemap["PixelFormat"].value = "Coord3D_C16"

        # Operating mode (distance range)
        mode_value = OPERATING_MODES.get(self.operating_mode)
        if mode_value is None:
            raise ValueError(
                f"Unknown operating_mode '{self.operating_mode}'. "
                f"Valid options: {list(OPERATING_MODES.keys())}"
            )
        nodemap["Scan3dOperatingMode"].value = mode_value
        logger.info("Operating mode: %s", mode_value)

        # Exposure time selector
        exp_value = EXPOSURE_SELECTORS.get(self.exposure)
        if exp_value is None:
            raise ValueError(
                f"Unknown exposure '{self.exposure}'. "
                f"Valid options: {list(EXPOSURE_SELECTORS.keys())}"
            )
        nodemap["ExposureTimeSelector"].value = exp_value
        logger.info("Exposure: %s", exp_value)

        # Conversion gain
        nodemap["ConversionGain"].value = self.conversion_gain

        # Image accumulation (frame averaging on-camera)
        nodemap["Scan3dImageAccumulation"].value = self.image_accumulation

        # Spatial filter (bilateral smoothing on-camera)
        nodemap["Scan3dSpatialFilterEnable"].value = self.spatial_filter

        # Confidence threshold (reject low-quality pixels on-camera)
        nodemap["Scan3dConfidenceThresholdEnable"].value = self.confidence_threshold

        logger.info(
            "Helios2 configured: gain=%s, accumulation=%d, "
            "spatial_filter=%s, confidence_threshold=%s",
            self.conversion_gain,
            self.image_accumulation,
            self.spatial_filter,
            self.confidence_threshold,
        )

    def _read_coordinate_params(self) -> None:
        """Read the scale and offset needed to convert raw uint16 Z values to metres."""
        nodemap = self._device.nodemap

        # Select the Z coordinate (CoordinateC) to read its scale and offset
        nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
        self._scale_z = float(nodemap["Scan3dCoordinateScale"].value)
        self._offset_z = float(nodemap["Scan3dCoordinateOffset"].value)

        logger.info(
            "Z coordinate params: scale=%.6f, offset=%.3f",
            self._scale_z,
            self._offset_z,
        )

    def _start_stream(self) -> None:
        """Configure transport layer and begin streaming."""
        tl = self._device.tl_stream_nodemap
        tl["StreamAutoNegotiatePacketSize"].value = self.stream_packet_negotiate
        tl["StreamPacketResendEnable"].value = self.stream_packet_resend
        tl["StreamBufferHandlingMode"].value = "NewestOnly"

        self._device.start_stream(1)
        self._streaming = True
        logger.info("Stream started (NewestOnly, 1 buffer)")

    def _buffer_to_depth_m(self, buffer: Any) -> np.ndarray:
        """
        Convert a Coord3D_C16 buffer to a (H, W) float32 depth array in metres.

        Coord3D_C16 is a single-channel 16-bit unsigned format where each
        pixel is a raw Z value. Invalid pixels are set to 0xFFFF (65535)
        by the camera.

        Conversion: depth_mm = raw_z * scale_z + offset_z
        The scale is already in mm, so we divide by 1000 to get metres.
        """
        h = buffer.height
        w = buffer.width

        # Cast the raw byte pointer to uint16
        pdata_uint16 = ctypes.cast(
            buffer.pdata, ctypes.POINTER(ctypes.c_uint16)
        )

        # Build a NumPy array from the ctypes pointer (zero-copy view)
        raw_z = np.ctypeslib.as_array(pdata_uint16, shape=(h * w,))
        raw_z = raw_z.reshape((h, w)).copy()  # copy to decouple from buffer

        # Mark invalid pixels as NaN.
        # The sensor uses 0xFFFF (65535) for out-of-range / no-return pixels
        # and 0 for pixels it could not measure. Both are invalid.
        depth_mm = raw_z.astype(np.float32)
        invalid_mask = (raw_z == 0xFFFF) | (raw_z == 0)
        depth_mm[invalid_mask] = np.nan

        # Apply scale and offset to get millimetres, then convert to metres
        depth_mm = depth_mm * self._scale_z + self._offset_z
        depth_m = depth_mm / 1000.0

        return depth_m
