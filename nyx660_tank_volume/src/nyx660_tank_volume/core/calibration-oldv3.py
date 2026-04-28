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
) -> np.ndarray:
    """
    Wide-angle corrected pixel area map.

    For a downward-facing wide-angle camera over a flat surface,
    pixels at the edge of the frame represent more physical area
    than pixels at the centre. The correction factor is 1/cos³(θ)
    where θ is the angle from the optical axis to each pixel.

    This method:
    1. Computes the off-axis angle θ for every pixel based on the
       camera's horizontal and vertical FoV
    2. Weights each pixel by 1/cos³(θ) (the wide-angle area correction)
    3. Normalises so the total area of all valid pixels equals the
       known physical tank floor area

    This produces accurate volume measurements at any position in the
    frame — centre, edge, or corner — without needing lens-specific
    intrinsic calibration.
    """
    h, w = valid_mask.shape

    # Compute the angle from optical axis for each pixel
    # Pixel coordinates normalised to [-1, 1] from centre
    cx, cy = w / 2.0, h / 2.0
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    nx = (xx - cx) / cx  # -1 to +1 across width
    ny = (yy - cy) / cy  # -1 to +1 across height

    # Convert normalised pixel position to angle from optical axis
    # At the edge of the frame, the angle equals half the FoV
    half_hfov_rad = np.radians(hfov_deg / 2.0)
    half_vfov_rad = np.radians(vfov_deg / 2.0)

    # Angle in each axis (using equidistant projection model,
    # which is common for wide-angle ToF cameras)
    theta_x = nx * half_hfov_rad
    theta_y = ny * half_vfov_rad

    # Total off-axis angle
    theta = np.sqrt(theta_x ** 2 + theta_y ** 2)

    # Wide-angle area correction: 1/cos³(θ)
    # This accounts for:
    #   1/cos(θ) — the projected pixel footprint stretches
    #   1/cos²(θ) — the distance to the surface increases at oblique angles
    cos_theta = np.cos(theta)
    # Clamp to avoid division by zero at extreme angles
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

        # If crop is enabled, scale the FoV to match the cropped region
        if cfg.camera.crop.enabled:
            crop = cfg.camera.crop
            crop_w = crop.x_max - crop.x_min + 1
            crop_h = crop.y_max - crop.y_min + 1
            # The crop region represents a fraction of the full FoV
            # Compute the angular span of the cropped region
            full_w = cfg.camera.width
            full_h = cfg.camera.height
            # Centre offset of crop within full frame
            crop_cx = (crop.x_min + crop.x_max) / 2.0
            crop_cy = (crop.y_min + crop.y_max) / 2.0
            # Fraction of full frame that the crop spans
            frac_w = crop_w / full_w
            frac_h = crop_h / full_h
            hfov = hfov * frac_w
            vfov = vfov * frac_h

        pixel_area = build_pixel_area_map_wideangle(
            valid_mask, cfg.camera.known_tank_area_m2, hfov, vfov
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
