"""
Mock backend mimicking the Helios2 Wide output characteristics.

Produces 640x480 depth frames with realistic noise, FoV coverage,
and edge falloff matching the Helios2 Wide's 108x78 degree FoV.
Use this to validate the full stack before the real sensor arrives.

No additional dependencies beyond numpy (already in requirements.txt).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from .base import CameraBackend, DepthFrame

logger = logging.getLogger(__name__)

# Helios2 Wide specs used for realistic mock output
HELIOS2_WIDE_HFOV_DEG = 108.0
HELIOS2_WIDE_VFOV_DEG = 78.0
HELIOS2_WIDE_MIN_RANGE_M = 0.3
HELIOS2_WIDE_MAX_RANGE_M = 8.3
HELIOS2_WIDE_INVALID_VALUE = float("nan")


class MockHelios2WideCamera(CameraBackend):
    """
    Simulates a Helios2 Wide mounted above a tank looking straight down.

    Generates synthetic depth maps with:
    - Correct 640x480 resolution
    - Realistic sensor noise (~2-4 mm std dev)
    - Edge pixel dropout (simulating wide-angle FoV falloff)
    - Configurable empty vs filled tank modes
    - NaN values for invalid/out-of-range pixels (matching real sensor)

    Use set_mode("empty") before calibration and set_mode("filled")
    before measurement to simulate the real workflow.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        mount_height_m: float = 2.5,
        tank_depth_m: float = 1.22,
        operating_mode: str = "5000mm",
        **kwargs,
    ) -> None:
        self.width = width
        self.height = height
        self.mount_height_m = mount_height_m
        self.tank_depth_m = tank_depth_m
        self.operating_mode = operating_mode
        self.rng = np.random.default_rng(42)
        self.opened = False
        self.mode = "filled"
        self._frame_count = 0

    def open(self) -> None:
        self.opened = True
        logger.info(
            "MockHelios2Wide opened: %dx%d, mount=%.2fm, tank_depth=%.2fm",
            self.width,
            self.height,
            self.mount_height_m,
            self.tank_depth_m,
        )

    def close(self) -> None:
        self.opened = False
        logger.info("MockHelios2Wide closed")

    def set_mode(self, mode: str) -> None:
        """Switch between 'empty' and 'filled' for testing."""
        if mode not in ("empty", "filled"):
            raise ValueError(f"Mode must be 'empty' or 'filled', got '{mode}'")
        self.mode = mode
        logger.info("MockHelios2Wide mode set to '%s'", mode)

    def get_frame(self) -> DepthFrame:
        if not self.opened:
            raise RuntimeError("Camera is not open")

        self._frame_count += 1

        # Base: distance from sensor to tank floor (empty tank)
        base_depth = self.mount_height_m

        # Create the empty tank floor with slight bowl distortion
        # (simulates wide-angle lens radial depth variation)
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        cy, cx = self.height / 2.0, self.width / 2.0
        r_norm = np.sqrt(
            ((xx - cx) / (self.width / 2.0)) ** 2
            + ((yy - cy) / (self.height / 2.0)) ** 2
        )

        # Wide-angle lenses measure slightly further at the edges
        lens_distortion = 0.015 * r_norm**2
        floor = np.full(
            (self.height, self.width), base_depth, dtype=np.float32
        ) + lens_distortion.astype(np.float32)

        if self.mode == "empty":
            depth = floor.copy()
        else:
            depth = self._generate_filled_scene(floor, xx, yy)

        # Sensor noise: ~3mm std dev, matching Helios2 Wide precision
        noise = self.rng.normal(0.0, 0.003, size=depth.shape).astype(
            np.float32
        )
        depth = depth + noise

        # Edge dropout: pixels beyond ~95% of the FoV radius get NaN
        # (simulates real sensor confidence falloff at extreme angles)
        edge_mask = r_norm > 0.95
        dropout_prob = np.clip((r_norm - 0.95) / 0.05, 0.0, 0.8)
        random_dropout = self.rng.random(size=depth.shape) < dropout_prob
        depth[edge_mask & random_dropout] = np.nan

        # Clamp to valid range
        depth = np.where(
            (depth >= HELIOS2_WIDE_MIN_RANGE_M)
            & (depth <= HELIOS2_WIDE_MAX_RANGE_M),
            depth,
            np.nan,
        )

        # Occasional random invalid pixels (~0.5% — simulates dust, etc.)
        random_invalid = self.rng.random(size=depth.shape) < 0.005
        depth[random_invalid] = np.nan

        return DepthFrame(
            depth_m=depth,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            rgb=None,
        )

    def _generate_filled_scene(
        self,
        floor: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a realistic drill cuttings surface.

        Creates an uneven surface with:
        - A primary mound (as if cuttings were dumped from one point)
        - A secondary ridge (as if a conveyor deposited a line of material)
        - Small-scale surface roughness (crumbly texture)
        """
        cx, cy = self.width * 0.45, self.height * 0.40
        rx, ry = self.width * 0.25, self.height * 0.20

        # Primary mound — off-center, like a dump point
        mound = 0.55 * np.exp(
            -(
                ((xx - cx) ** 2) / (2 * rx**2)
                + ((yy - cy) ** 2) / (2 * ry**2)
            )
        )

        # Secondary ridge — linear feature across the tank
        ridge_center = self.height * 0.65
        ridge_width = self.height * 0.06
        ridge = 0.25 * np.exp(
            -((yy - ridge_center) ** 2) / (2 * ridge_width**2)
        )

        # Small scattered clumps
        clump1_x, clump1_y = self.width * 0.7, self.height * 0.3
        clump1 = 0.15 * np.exp(
            -(
                ((xx - clump1_x) ** 2) / (2 * (self.width * 0.08) ** 2)
                + ((yy - clump1_y) ** 2) / (2 * (self.height * 0.06) ** 2)
            )
        )

        # Combine and limit to tank depth
        fill_height = np.clip(mound + ridge + clump1, 0.0, self.tank_depth_m * 0.85)

        # Surface roughness — small-scale texture like crumbly cuttings
        roughness = self.rng.normal(0.0, 0.008, size=floor.shape).astype(
            np.float32
        )
        fill_height = np.clip(
            fill_height.astype(np.float32) + roughness, 0.0, None
        )

        # Depth = floor distance minus fill height
        return floor - fill_height

    @property
    def is_mock(self) -> bool:
        return True
