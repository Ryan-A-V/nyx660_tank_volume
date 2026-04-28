from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from nyx660_tank_volume.config import AppConfig
from nyx660_tank_volume.core.tank_detect import detect_tank_floor
from nyx660_tank_volume.utils.io import load_npz, save_npz

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBundle:
    baseline_depth_m: np.ndarray
    valid_mask: np.ndarray
    pixel_area_m2: np.ndarray
    created_utc: str
    footprint_area_m2: float
    reference_integral_m3: float
    detection_info: dict  # auto-detect metadata


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
            detection_info={},
        )


def build_pixel_area_map_intrinsics(
    baseline_depth_m: np.ndarray, fx: float, fy: float
) -> np.ndarray:
    """Pinhole model pixel area. For narrow FoV sensors."""
    return (baseline_depth_m ** 2) / (fx * fy)


def build_pixel_area_map_wideangle(
    floor_mask: np.ndarray,
    known_tank_area_m2: float,
    hfov_deg: float,
    vfov_deg: float,
    full_width: int,
    full_height: int,
) -> np.ndarray:
    """
    Wide-angle corrected pixel area map using the floor mask.

    Unlike the crop-based version, this uses the auto-detected floor
    mask directly. Each pixel's angle is computed from its position
    in the full sensor frame relative to the optical centre.

    The 1/cos³(θ) correction accounts for the wide-angle projection
    where edge pixels represent more physical area than centre pixels.
    The total is normalised to the known tank floor area.
    """
    h, w = floor_mask.shape

    full_cx = full_width / 2.0
    full_cy = full_height / 2.0

    half_hfov_rad = np.radians(hfov_deg / 2.0)
    half_vfov_rad = np.radians(vfov_deg / 2.0)

    yy, xx = np.mgrid[0:h, 0:w]

    # If the mask is smaller than the full frame (due to crop),
    # we still compute angles relative to the full frame centre.
    # However, with auto-detect the mask IS the full frame size,
    # just with False outside the tank.
    nx = (xx - full_cx) / (full_width / 2.0)
    ny = (yy - full_cy) / (full_height / 2.0)

    theta_x = nx * half_hfov_rad
    theta_y = ny * half_vfov_rad
    theta = np.sqrt(theta_x ** 2 + theta_y ** 2)

    cos_theta = np.cos(theta)
    cos_theta = np.clip(cos_theta, 0.1, 1.0)
    weight = 1.0 / (cos_theta ** 3)

    # Only apply to floor pixels
    weight = np.where(floor_mask, weight, 0.0)

    total_weight = np.sum(weight)
    if total_weight == 0:
        return np.zeros((h, w), dtype=np.float32)

    pixel_area = (weight / total_weight) * known_tank_area_m2
    return pixel_area.astype(np.float32)


def create_calibration(
    depth_frames_m: list[np.ndarray], cfg: AppConfig
) -> CalibrationBundle:
    """
    Create a calibration from depth frames of an empty tank.

    Calibration flow:
    1. Stack frames and compute median baseline depth
    2. Build a consistency mask — only keep pixels that returned
       valid depth in at least 90% of calibration frames
    3. If auto_detect is enabled: detect the tank floor and use
       the floor mask intersected with the consistency mask
    4. If manual crop: use crop box intersected with consistency mask
    5. Compute per-pixel area
    6. Bundle everything and return
    """
    stack = np.stack(depth_frames_m, axis=0).astype(np.float32)
    baseline_depth = np.nanmedian(stack, axis=0)

    # Basic valid mask from depth range
    range_mask = np.isfinite(baseline_depth)
    range_mask &= baseline_depth >= cfg.measurement.min_valid_depth_m
    range_mask &= baseline_depth <= cfg.measurement.max_valid_depth_m

    # Consistency mask: only keep pixels that were valid in >= 90%
    # of calibration frames. This excludes flaky edge pixels that
    # return data intermittently — they would cause quality drops
    # during measurement when they randomly go NaN.
    valid_per_frame = np.isfinite(stack)
    valid_fraction = np.mean(valid_per_frame, axis=0)
    consistency_threshold = 0.90
    consistency_mask = valid_fraction >= consistency_threshold

    range_mask &= consistency_mask

    logger.info(
        "Consistency filter: %d/%d pixels passed %.0f%% threshold (removed %d flaky edge pixels)",
        int(np.sum(consistency_mask)),
        consistency_mask.size,
        consistency_threshold * 100,
        int(np.sum(~consistency_mask & np.any(valid_per_frame, axis=0))),
    )

    detection_info: dict = {}

    # ----- Determine the floor mask -----

    floor_mask = None

    # Try auto-detection first
    if cfg.auto_detect.enabled:
        logger.info("Running automatic tank floor detection...")

        # Determine known tank area for validation
        detect_tank_area = 0.0
        if cfg.tank is not None:
            detect_tank_area = cfg.tank.floor_area_m2
        elif cfg.camera.known_tank_area_m2 is not None:
            detect_tank_area = cfg.camera.known_tank_area_m2

        result = detect_tank_floor(
            baseline_depth_m=baseline_depth,
            mount_height_m=cfg.camera.mount_height_m,
            known_tank_area_m2=detect_tank_area,
            floor_tolerance_m=cfg.auto_detect.floor_tolerance_m,
            min_floor_fraction=cfg.auto_detect.min_floor_fraction,
            morphology_kernel=cfg.auto_detect.morphology_kernel,
            min_valid_depth_m=cfg.measurement.min_valid_depth_m,
            max_valid_depth_m=cfg.measurement.max_valid_depth_m,
        )

        detection_info = {
            "method": "auto_detect",
            "success": result.success,
            "message": result.message,
            "warnings": result.warnings,
            "floor_depth_m": result.floor_depth_m,
            "floor_pixel_count": result.floor_pixel_count,
            "floor_fraction": result.floor_fraction,
            "detected_bounds": result.detected_bounds,
        }

        if result.success:
            floor_mask = result.floor_mask & range_mask & consistency_mask
            logger.info(
                "Auto-detect succeeded: %d floor pixels (after consistency filter), depth=%.3f m",
                int(np.sum(floor_mask)),
                result.floor_depth_m,
            )
        else:
            logger.warning(
                "Auto-detect failed: %s. Falling back to manual crop.",
                result.message,
            )

    # Fall back to manual crop if auto-detect is off or failed
    if floor_mask is None:
        if cfg.camera.crop.enabled:
            logger.info("Using manual crop box for floor mask")
            crop = cfg.camera.crop
            crop_mask = np.zeros_like(range_mask)
            crop_mask[crop.y_min : crop.y_max + 1, crop.x_min : crop.x_max + 1] = True
            floor_mask = range_mask & crop_mask
            detection_info["method"] = "manual_crop"
            detection_info["crop"] = {
                "x_min": crop.x_min,
                "x_max": crop.x_max,
                "y_min": crop.y_min,
                "y_max": crop.y_max,
            }
        else:
            # No crop, no auto-detect — use all valid pixels
            logger.info("No crop or auto-detect — using all valid pixels")
            floor_mask = range_mask
            detection_info["method"] = "full_frame"

    valid_mask = floor_mask

    # ----- Compute pixel area -----

    # Determine the known tank area
    known_area = None
    if cfg.camera.known_tank_area_m2 is not None:
        known_area = cfg.camera.known_tank_area_m2
    elif cfg.tank is not None:
        known_area = cfg.tank.floor_area_m2

    if known_area is not None:
        # Wide-angle corrected area using floor mask
        pixel_area = build_pixel_area_map_wideangle(
            floor_mask=valid_mask,
            known_tank_area_m2=known_area,
            hfov_deg=108.0,
            vfov_deg=78.0,
            full_width=cfg.camera.width,
            full_height=cfg.camera.height,
        )
        footprint_area = known_area

    elif cfg.camera.intrinsics is not None:
        # Pinhole intrinsics mode
        intr = cfg.camera.intrinsics
        pixel_area = build_pixel_area_map_intrinsics(
            baseline_depth, intr.fx, intr.fy
        )
        pixel_area = np.where(valid_mask, pixel_area, 0.0)
        footprint_area = float(np.sum(pixel_area[valid_mask]))

    else:
        raise ValueError(
            "Cannot compute pixel area. Provide one of: "
            "tank dimensions (tank.length_m, tank.width_m), "
            "camera.known_tank_area_m2, or camera.intrinsics."
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
        detection_info=detection_info,
    )
