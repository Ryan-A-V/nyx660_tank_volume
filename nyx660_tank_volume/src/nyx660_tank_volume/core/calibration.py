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


def build_pixel_area_map(baseline_depth_m: np.ndarray, fx: float, fy: float) -> np.ndarray:
    # Approximate world area represented by each pixel on a plane parallel to the camera.
    return (baseline_depth_m**2) / (fx * fy)


def create_calibration(depth_frames_m: list[np.ndarray], cfg: AppConfig) -> CalibrationBundle:
    stack = np.stack(depth_frames_m, axis=0).astype(np.float32)
    baseline_depth = np.nanmedian(stack, axis=0)
    valid_mask = np.isfinite(baseline_depth)
    valid_mask &= baseline_depth >= cfg.measurement.min_valid_depth_m
    valid_mask &= baseline_depth <= cfg.measurement.max_valid_depth_m

    intr = cfg.camera.intrinsics
    pixel_area = build_pixel_area_map(baseline_depth, intr.fx, intr.fy)
    pixel_area = np.where(valid_mask, pixel_area, 0.0)

    footprint_area = float(np.sum(pixel_area[valid_mask]))
    reference_integral = float(np.sum(baseline_depth[valid_mask] * pixel_area[valid_mask]))

    return CalibrationBundle(
        baseline_depth_m=baseline_depth,
        valid_mask=valid_mask,
        pixel_area_m2=pixel_area,
        created_utc=datetime.now(timezone.utc).isoformat(),
        footprint_area_m2=footprint_area,
        reference_integral_m3=reference_integral,
    )
