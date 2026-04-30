from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from nyx660_tank_volume.config import AppConfig
from nyx660_tank_volume.core.calibration import CalibrationBundle


@dataclass
class MeasurementResult:
    timestamp_utc: str
    estimated_volume_m3: float
    estimated_volume_liters: float | None
    relative_fill_ratio: float | None
    occupied_surface_area_m2: float
    average_fill_height_m: float
    max_fill_height_m: float
    valid_pixel_ratio: float
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess_depth(depth_m: np.ndarray, cfg: AppConfig) -> np.ndarray:
    arr = depth_m.astype(np.float32).copy()
    c = cfg.camera.crop
    if c.enabled:
        arr = arr[c.y_min : c.y_max + 1, c.x_min : c.x_max + 1]
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < cfg.measurement.min_valid_depth_m) | (arr > cfg.measurement.max_valid_depth_m)] = np.nan
    return arr


def smooth_depth(frames: list[np.ndarray]) -> np.ndarray:
    return np.nanmedian(np.stack(frames, axis=0).astype(np.float32), axis=0)


def _build_volume_correction_map(h: int, w: int, full_width: int, full_height: int) -> np.ndarray:
    """
    Build a per-pixel volume correction factor based on radial
    distance from the optical centre.

    The cos^3 area model distributes the total tank area correctly
    but over-assigns area to centre pixels and under-assigns to
    edge pixels. This correction factor is applied to each pixel's
    volume contribution (delta_h * pixel_area) AFTER the area
    calculation, so normalisation cannot cancel it out.

    Empirically measured with a 4'x4' plywood sheet at known positions:
        dist 0.19 (centre): area was 1.671x too high -> correction 0.598
        dist 0.90 (corner): area was 0.947x -> correction 1.056

    The correction curve interpolates between these measured points.
    """
    full_cx = full_width / 2.0
    full_cy = full_height / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - full_cx) / (full_width / 2.0)
    ny = (yy - full_cy) / (full_height / 2.0)
    dist = np.sqrt(nx ** 2 + ny ** 2)

    # Empirical correction factors at measured distances
    # These correct the cos^3 area model's known errors
    cal_dist =       np.array([0.00, 0.19, 0.40, 0.61, 0.82, 0.90, 0.95, 1.20])
    cal_correction = np.array([0.44, 0.44, 0.68, 1.00, 1.15, 1.21, 1.21, 1.21])

    correction = np.interp(dist, cal_dist, cal_correction)
    return correction.astype(np.float32)


def estimate_volume(current_depth_m: np.ndarray, calib: CalibrationBundle, cfg: AppConfig) -> MeasurementResult:
    notes: list[str] = []
    current = current_depth_m.astype(np.float32)

    # valid = pixels that are in the floor mask AND have valid depth now
    valid = calib.valid_mask & np.isfinite(current)

    # Quality: what fraction of FLOOR pixels returned valid data
    floor_pixel_count = float(np.sum(calib.valid_mask))
    valid_floor_count = float(np.sum(valid))
    valid_pixel_ratio = (
        valid_floor_count / floor_pixel_count
        if floor_pixel_count > 0
        else 0.0
    )

    delta_h = calib.baseline_depth_m - current
    delta_h = np.where(valid, delta_h, 0.0)

    low = np.nanpercentile(delta_h[valid], cfg.measurement.outlier_clip_percentile_low) if np.any(valid) else 0.0
    high = np.nanpercentile(delta_h[valid], cfg.measurement.outlier_clip_percentile_high) if np.any(valid) else 0.0
    delta_h = np.clip(delta_h, low, high)
    delta_h = np.clip(delta_h, 0.0, cfg.measurement.max_height_step_m)
    delta_h[delta_h < cfg.measurement.fill_threshold_m] = 0.0

    # Per-pixel volume with position-dependent correction
    h, w = current.shape
    correction_map = _build_volume_correction_map(
        h, w, cfg.camera.width, cfg.camera.height
    )
    per_pixel_volume = delta_h * calib.pixel_area_m2 * correction_map

    est_volume_m3 = float(np.sum(per_pixel_volume))

    # Occupied area also needs correction for consistency
    corrected_area = calib.pixel_area_m2 * correction_map
    occupied_area = float(np.sum(corrected_area[delta_h > cfg.measurement.occupancy_mask_min_height_m]))
    avg_height = est_volume_m3 / occupied_area if occupied_area > 0 else 0.0
    max_height = float(np.nanmax(delta_h)) if np.any(delta_h > 0) else 0.0

    est_liters = None
    relative_fill_ratio = None
    if cfg.measurement.known_volume_liters is not None and calib.footprint_area_m2 > 0:
        est_liters = est_volume_m3 * 1000.0
        relative_fill_ratio = min(est_liters / cfg.measurement.known_volume_liters, 1.0)
        notes.append("Relative fill ratio uses configured known_volume_liters.")
    else:
        notes.append(
            "Absolute liters require measurement.known_volume_liters. Returning volume from empty-tank baseline geometry only."
        )

    if valid_pixel_ratio < 0.60:
        notes.append("Low valid pixel ratio; check exposure, mounting, or tank surface reflectivity.")

    return MeasurementResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        estimated_volume_m3=est_volume_m3,
        estimated_volume_liters=est_liters,
        relative_fill_ratio=relative_fill_ratio,
        occupied_surface_area_m2=occupied_area,
        average_fill_height_m=avg_height,
        max_fill_height_m=max_height,
        valid_pixel_ratio=valid_pixel_ratio,
        notes=notes,
    )
