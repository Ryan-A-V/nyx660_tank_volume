from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from nyx660_tank_volume.config import AppConfig
from nyx660_tank_volume.utils.io import load_npz, save_npz


@dataclass
class CalibrationBundle:
    baseline_depth_m: np.ndarray
    valid_mask: np.ndarray
    pixel_area_m2: np.ndarray
    created_utc: str
    footprint_area_m2: float
    reference_integral_m3: float


class CalibrationStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def exists(self) -> bool:
        try:
            load_npz(self.path)
            return True
        except FileNotFoundError:
            return False

    def save(self, bundle: CalibrationBundle) -> None:
        save_npz(
            self.path,
            baseline_depth_m=bundle.baseline_depth_m.astype(np.float32),
            valid_mask=bundle.valid_mask.astype(np.uint8),
            pixel_area_m2=bundle.pixel_area_m2.astype(np.float32),
            created_utc=np.array(bundle.created_utc),
            footprint_area_m2=np.array(bundle.footprint_area_m2, dtype=np.float32),
            reference_integral_m3=np.array(bundle.reference_integral_m3, dtype=np.float32),
        )

    def load(self) -> CalibrationBundle:
        data = load_npz(self.path)
        return CalibrationBundle(
            baseline_depth_m=data["baseline_depth_m"].astype(np.float32),
            valid_mask=data["valid_mask"].astype(bool),
            pixel_area_m2=data["pixel_area_m2"].astype(np.float32),
            created_utc=str(data["created_utc"].tolist()),
            footprint_area_m2=float(data["footprint_area_m2"]),
            reference_integral_m3=float(data["reference_integral_m3"]),
        )


def build_pixel_area_map_intrinsics(
    baseline_depth_m: np.ndarray, fx: float, fy: float
) -> np.ndarray:
    """
    Pinhole model pixel area.
    Works well for narrow FoV sensors but inaccurate at wide angles.
    """
    return (baseline_depth_m ** 2) / (fx * fy)


def build_pixel_area_map_wideangle(
    valid_mask: np.ndarray,
    known_tank_area_m2: float,
    hfov_deg: float,
    vfov_deg: float,
    full_width: int,
    full_height: int,
    crop_x_min: int = 0,
    crop_y_min: int = 0,
) -> np.ndarray:
    """
    Wide-angle corrected pixel area map.

    For a downward-facing wide-angle camera over a flat surface,
    pixels at the edge of the frame represent more physical area
    than pixels at the centre. The correction factor is 1/cos³(θ)
    where θ is the angle from the optical axis to each pixel.

    IMPORTANT: Pixel angles are computed relative to the full
    sensor frame, not the cropped region. Cropping selects a subset
    of pixels but does not change their angular positions in the
    lens projection. The crop offsets map each cropped pixel back
    to its position in the full frame before computing its angle.
    """
    crop_h, crop_w = valid_mask.shape

    # Full frame centre (optical axis)
    full_cx = full_width / 2.0
    full_cy = full_height / 2.0

    # Half-angle per pixel in the full frame
    half_hfov_rad = np.radians(hfov_deg / 2.0)
    half_vfov_rad = np.radians(vfov_deg / 2.0)

    # Build pixel coordinate grids in the cropped frame
    xx_crop, yy_crop = np.meshgrid(
        np.arange(crop_w), np.arange(crop_h)
    )

    # Map cropped pixel positions back to full frame positions
    xx_full = xx_crop + crop_x_min
    yy_full = yy_crop + crop_y_min

    # Normalise to [-1, 1] relative to full frame centre
    nx = (xx_full - full_cx) / (full_width / 2.0)
    ny = (yy_full - full_cy) / (full_height / 2.0)

    # Angle from optical axis per pixel (equidistant projection)
    theta_x = nx * half_hfov_rad
    theta_y = ny * half_vfov_rad
    theta = np.sqrt(theta_x ** 2 + theta_y ** 2)

    # Wide-angle area correction: 1/cos³(θ)
    cos_theta = np.cos(theta)
    cos_theta = np.clip(cos_theta, 0.1, 1.0)
    weight = 1.0 / (cos_theta ** 3)

    # Zero out invalid pixels
    weight = np.where(valid_mask, weight, 0.0)

    # Normalise so total area equals known tank area
    total_weight = np.sum(weight)
    if total_weight == 0:
        return np.zeros_like(valid_mask, dtype=np.float32)

    pixel_area = (weight / total_weight) * known_tank_area_m2
    return pixel_area.astype(np.float32)


def create_calibration(
    depth_frames_m: list[np.ndarray], cfg: AppConfig
) -> CalibrationBundle:
    stack = np.stack(depth_frames_m, axis=0).astype(np.float32)
    baseline_depth = np.nanmedian(stack, axis=0)
    valid_mask = np.isfinite(baseline_depth)
    valid_mask &= baseline_depth >= cfg.measurement.min_valid_depth_m
    valid_mask &= baseline_depth <= cfg.measurement.max_valid_depth_m

    # Choose pixel area calculation method
    if cfg.camera.known_tank_area_m2 is not None:
        # Wide-angle corrected mode — recommended for Helios2 Wide
        hfov = 108.0  # Helios2 Wide horizontal FoV
        vfov = 78.0   # Helios2 Wide vertical FoV

        # Pass full frame dimensions and crop offset so pixel angles
        # are computed relative to the optical axis, not the crop centre
        crop_x_min = 0
        crop_y_min = 0
        if cfg.camera.crop.enabled:
            crop_x_min = cfg.camera.crop.x_min
            crop_y_min = cfg.camera.crop.y_min

        pixel_area = build_pixel_area_map_wideangle(
            valid_mask,
            cfg.camera.known_tank_area_m2,
            hfov,
            vfov,
            full_width=cfg.camera.width,
            full_height=cfg.camera.height,
            crop_x_min=crop_x_min,
            crop_y_min=crop_y_min,
        )
        footprint_area = cfg.camera.known_tank_area_m2

    elif cfg.camera.intrinsics is not None:
        # Pinhole intrinsics mode — for narrow FoV sensors
        intr = cfg.camera.intrinsics
        pixel_area = build_pixel_area_map_intrinsics(
            baseline_depth, intr.fx, intr.fy
        )
        pixel_area = np.where(valid_mask, pixel_area, 0.0)
        footprint_area = float(np.sum(pixel_area[valid_mask]))

    else:
        raise ValueError(
            "Either camera.known_tank_area_m2 or camera.intrinsics "
            "must be set in the config."
        )

    reference_integral = float(
        np.sum(baseline_depth[valid_mask] * pixel_area[valid_mask])
    )

    return CalibrationBundle(
        baseline_depth_m=baseline_depth,
        valid_mask=valid_mask,
        pixel_area_m2=pixel_area,
        created_utc=datetime.now(timezone.utc).isoformat(),
        footprint_area_m2=footprint_area,
        reference_integral_m3=reference_integral,
    )
