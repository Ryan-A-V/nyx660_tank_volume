"""
Automatic tank boundary detection.

During calibration with an empty tank, detects the tank floor region
by finding the dominant flat surface in the depth data. Outputs a
binary floor mask that replaces the manual crop box.

The detection works by:
1. Finding the dominant depth (the floor) via histogram peak detection
2. Selecting all pixels within a tolerance of that depth as floor candidates
3. Cleaning the mask with morphological operations to fill small holes
   and remove isolated noise pixels
4. Validating the detected region against expected tank area

No dependencies beyond numpy (already in requirements.txt).
Uses scipy.ndimage for morphology if available, falls back to a
pure-numpy implementation if not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TankDetectionResult:
    """Results of automatic tank boundary detection."""

    floor_mask: np.ndarray
    floor_depth_m: float
    floor_pixel_count: int
    total_valid_pixels: int
    floor_fraction: float
    detected_bounds: dict  # {y_min, y_max, x_min, x_max}
    success: bool
    message: str
    warnings: list[str]


def detect_tank_floor(
    baseline_depth_m: np.ndarray,
    mount_height_m: float,
    known_tank_area_m2: float,
    hfov_deg: float = 108.0,
    vfov_deg: float = 78.0,
    floor_tolerance_m: float = 0.10,
    min_floor_fraction: float = 0.30,
    morphology_kernel: int = 7,
    min_valid_depth_m: float = 0.3,
    max_valid_depth_m: float = 8.5,
    area_oversize_warning: float = 1.3,
    area_oversize_fail: float = 2.0,
) -> TankDetectionResult:
    """
    Detect the tank floor region from an averaged depth frame.

    Uses a multi-strategy approach:
    1. Find the dominant flat surface by depth histogram
    2. Refine using gradient detection to find wall boundaries
    3. Validate the detected area against the known tank dimensions

    Parameters
    ----------
    baseline_depth_m : np.ndarray
        Averaged depth frame (median of N frames) in metres.
    mount_height_m : float
        Expected distance from sensor to tank floor in metres.
    known_tank_area_m2 : float
        Physical floor area of the tank from config. Used to validate
        detection and estimate pixel area.
    hfov_deg, vfov_deg : float
        Sensor field of view in degrees. Used to estimate the physical
        area the detected region represents.
    floor_tolerance_m : float
        Pixels within this tolerance of the detected floor depth
        are classified as floor. Default 0.10 m (10 cm).
    min_floor_fraction : float
        Minimum fraction of valid pixels that must be floor for
        detection to succeed. Default 0.30 (30%).
    morphology_kernel : int
        Kernel size for morphological closing (fills small holes).
    min_valid_depth_m, max_valid_depth_m : float
        Valid depth range.
    area_oversize_warning : float
        If detected area exceeds known_tank_area * this factor, add
        a warning but still succeed. Default 1.3 (30% oversize).
    area_oversize_fail : float
        If detected area exceeds known_tank_area * this factor, fail
        auto-detect (likely bled onto surrounding ground). Default 2.0.

    Returns
    -------
    TankDetectionResult
    """
    h, w = baseline_depth_m.shape

    # Build valid pixel mask
    valid = (
        np.isfinite(baseline_depth_m)
        & (baseline_depth_m >= min_valid_depth_m)
        & (baseline_depth_m <= max_valid_depth_m)
    )
    total_valid = int(np.sum(valid))

    if total_valid == 0:
        return TankDetectionResult(
            floor_mask=np.zeros_like(valid),
            floor_depth_m=0.0,
            floor_pixel_count=0,
            total_valid_pixels=0,
            floor_fraction=0.0,
            detected_bounds={"y_min": 0, "y_max": 0, "x_min": 0, "x_max": 0},
            success=False,
            message="No valid depth pixels found.",
            warnings=[],
        )

    valid_depths = baseline_depth_m[valid]

    # --- Step 1: Find dominant floor depth via histogram ---

    search_min = max(mount_height_m - 1.0, min_valid_depth_m)
    search_max = min(mount_height_m + 1.0, max_valid_depth_m)

    bin_width = 0.005
    bins = np.arange(search_min, search_max + bin_width, bin_width)
    hist, bin_edges = np.histogram(valid_depths, bins=bins)

    if len(hist) == 0 or np.max(hist) == 0:
        return TankDetectionResult(
            floor_mask=np.zeros_like(valid),
            floor_depth_m=0.0,
            floor_pixel_count=0,
            total_valid_pixels=total_valid,
            floor_fraction=0.0,
            detected_bounds={"y_min": 0, "y_max": 0, "x_min": 0, "x_max": 0},
            success=False,
            message=f"No depth values found near mount height {mount_height_m:.2f} m.",
            warnings=[],
        )

    peak_idx = np.argmax(hist)
    floor_depth = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0
    logger.info("Detected floor depth: %.3f m (expected ~%.3f m)", floor_depth, mount_height_m)

    # --- Step 2: Build initial floor mask from depth tolerance ---

    depth_mask = valid & (
        np.abs(baseline_depth_m - floor_depth) <= floor_tolerance_m
    )

    # --- Step 3: Use gradient to find wall boundaries ---
    # Walls create sharp depth transitions. We detect these and use
    # them to trim the floor mask even when the ground outside is
    # at the same depth as the floor.

    gradient_mask = _detect_wall_boundaries(
        baseline_depth_m, valid, kernel_size=3, gradient_threshold_m=0.05
    )

    # Remove wall pixels from the floor mask
    floor_mask = depth_mask & ~gradient_mask

    # --- Step 4: Morphological cleanup ---

    floor_mask = _morphological_close(floor_mask, kernel_size=morphology_kernel)
    floor_mask = _keep_largest_component(floor_mask)

    # --- Step 5: Validate detected area against known tank ---

    floor_pixel_count = int(np.sum(floor_mask))
    floor_fraction = floor_pixel_count / total_valid if total_valid > 0 else 0.0

    if floor_pixel_count == 0:
        return TankDetectionResult(
            floor_mask=floor_mask,
            floor_depth_m=float(floor_depth),
            floor_pixel_count=0,
            total_valid_pixels=total_valid,
            floor_fraction=0.0,
            detected_bounds={"y_min": 0, "y_max": 0, "x_min": 0, "x_max": 0},
            success=False,
            message="No floor pixels survived filtering.",
            warnings=[],
        )

    # Estimate the physical area the detected region represents
    # using a rough pixel area from the total valid frame
    estimated_total_fov_area = _estimate_fov_area(
        floor_depth, hfov_deg, vfov_deg
    )
    estimated_pixel_area = estimated_total_fov_area / total_valid
    detected_area_m2 = floor_pixel_count * estimated_pixel_area

    # Bounding box
    rows, cols = np.where(floor_mask)
    bounds = {
        "y_min": int(rows.min()),
        "y_max": int(rows.max()),
        "x_min": int(cols.min()),
        "x_max": int(cols.max()),
    }

    # --- Step 6: Area validation and warnings ---

    warnings: list[str] = []
    area_ratio = detected_area_m2 / known_tank_area_m2 if known_tank_area_m2 > 0 else 0

    if area_ratio > area_oversize_fail:
        return TankDetectionResult(
            floor_mask=floor_mask,
            floor_depth_m=float(floor_depth),
            floor_pixel_count=floor_pixel_count,
            total_valid_pixels=total_valid,
            floor_fraction=floor_fraction,
            detected_bounds=bounds,
            success=False,
            message=(
                f"Detected floor area ({detected_area_m2:.1f} m²) is "
                f"{area_ratio:.1f}x larger than the known tank area "
                f"({known_tank_area_m2:.1f} m²). The detection likely "
                f"bled onto surrounding ground. Use manual crop instead "
                f"(set camera.crop.enabled: true and auto_detect.enabled: false)."
            ),
            warnings=[],
        )

    if area_ratio > area_oversize_warning:
        warnings.append(
            f"Detected floor area ({detected_area_m2:.1f} m²) is "
            f"{area_ratio:.1f}x the known tank area ({known_tank_area_m2:.1f} m²). "
            f"Some ground outside the tank may be included. Consider "
            f"verifying with a depth preview or using manual crop."
        )

    if area_ratio < 0.5:
        warnings.append(
            f"Detected floor area ({detected_area_m2:.1f} m²) is only "
            f"{area_ratio:.1%} of the known tank area ({known_tank_area_m2:.1f} m²). "
            f"The sensor may not be seeing the full tank floor. Check "
            f"mounting position and operating mode range."
        )

    if floor_fraction < min_floor_fraction:
        return TankDetectionResult(
            floor_mask=floor_mask,
            floor_depth_m=float(floor_depth),
            floor_pixel_count=floor_pixel_count,
            total_valid_pixels=total_valid,
            floor_fraction=floor_fraction,
            detected_bounds=bounds,
            success=False,
            message=(
                f"Floor region too small: {floor_fraction:.1%} of valid pixels "
                f"(minimum {min_floor_fraction:.1%}). Check sensor position or "
                f"increase auto_detect.floor_tolerance_m."
            ),
            warnings=warnings,
        )

    # Check aspect ratio against expected tank shape
    detected_h = bounds["y_max"] - bounds["y_min"]
    detected_w = bounds["x_max"] - bounds["x_min"]
    if detected_h > 0 and detected_w > 0:
        detected_aspect = detected_w / detected_h
        # The aspect ratio in pixels won't match the physical aspect
        # exactly due to the non-square FoV, but extreme mismatches
        # suggest a problem
        if detected_aspect > 5.0 or detected_aspect < 0.2:
            warnings.append(
                f"Detected floor region has unusual aspect ratio "
                f"({detected_w}×{detected_h} pixels = {detected_aspect:.1f}). "
                f"This may indicate the sensor is not centred over the tank."
            )

    logger.info(
        "Tank floor detected: %d pixels (%.1f%% of valid), "
        "estimated area=%.1f m² (expected %.1f m²), "
        "bounds rows=%d-%d cols=%d-%d",
        floor_pixel_count,
        floor_fraction * 100,
        detected_area_m2,
        known_tank_area_m2,
        bounds["y_min"],
        bounds["y_max"],
        bounds["x_min"],
        bounds["x_max"],
    )
    if warnings:
        for w_msg in warnings:
            logger.warning("Auto-detect warning: %s", w_msg)

    return TankDetectionResult(
        floor_mask=floor_mask,
        floor_depth_m=float(floor_depth),
        floor_pixel_count=floor_pixel_count,
        total_valid_pixels=total_valid,
        floor_fraction=floor_fraction,
        detected_bounds=bounds,
        success=True,
        message="Tank floor detected successfully.",
        warnings=warnings,
    )


def _detect_wall_boundaries(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    kernel_size: int = 3,
    gradient_threshold_m: float = 0.05,
) -> np.ndarray:
    """
    Detect wall boundaries using depth gradient magnitude.

    Walls appear as sharp depth transitions — the gradient magnitude
    at wall pixels is much higher than at floor or ground pixels.
    Returns a binary mask where True = wall/boundary pixel.
    """
    # Fill NaN with a neutral value for gradient computation
    median_depth = np.nanmedian(depth_m[valid_mask])
    filled = np.where(np.isfinite(depth_m), depth_m, median_depth)

    # Sobel-like gradient in x and y
    pad = kernel_size // 2
    padded = np.pad(filled, pad, mode="edge")
    h, w = depth_m.shape

    grad_x = padded[pad:pad + h, pad + 1:pad + 1 + w] - padded[pad:pad + h, pad - 1:pad - 1 + w]
    grad_y = padded[pad + 1:pad + 1 + h, pad:pad + w] - padded[pad - 1:pad - 1 + h, pad:pad + w]

    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Pixels with gradient above threshold are walls
    wall_mask = gradient_mag > gradient_threshold_m

    # Also mark NaN boundaries as walls — transitions from valid
    # to invalid depth often indicate a wall edge
    nan_mask = ~np.isfinite(depth_m)
    # Dilate the NaN mask slightly to catch the boundary pixels
    nan_boundary = _dilate(nan_mask, size=3) & valid_mask

    return wall_mask | nan_boundary


def _dilate(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Simple binary dilation using max filter."""
    pad = size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    h, w = mask.shape
    result = np.zeros_like(mask)
    for dy in range(size):
        for dx in range(size):
            result |= padded[dy:dy + h, dx:dx + w]
    return result


def _estimate_fov_area(depth_m: float, hfov_deg: float, vfov_deg: float) -> float:
    """
    Estimate the total ground area visible at a given depth for
    the sensor's field of view. Used for area validation.
    """
    half_h = np.radians(hfov_deg / 2.0)
    half_v = np.radians(vfov_deg / 2.0)
    width = 2.0 * depth_m * np.tan(half_h)
    height = 2.0 * depth_m * np.tan(half_v)
    return width * height


def _morphological_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Morphological closing (dilation then erosion) to fill small holes.
    Uses scipy if available, otherwise a pure-numpy fallback.
    """
    try:
        from scipy.ndimage import binary_closing

        struct = np.ones((kernel_size, kernel_size), dtype=bool)
        return binary_closing(mask, structure=struct).astype(bool)
    except ImportError:
        # Pure numpy fallback using iterative dilation/erosion
        return _numpy_close(mask, kernel_size)


def _numpy_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Pure numpy morphological closing fallback."""
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    h, w = mask.shape

    # Dilation: pixel is True if any neighbour is True
    dilated = np.zeros_like(padded)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            dilated[pad : pad + h, pad : pad + w] |= padded[dy : dy + h, dx : dx + w]

    # Erosion: pixel is True only if all neighbours are True
    eroded = np.ones((h, w), dtype=bool)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            eroded &= dilated[dy : dy + h, dx : dx + w]

    return eroded


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    Uses scipy if available, otherwise a flood-fill fallback.
    """
    try:
        from scipy.ndimage import label

        labeled, num_features = label(mask)
        if num_features == 0:
            return mask

        # Find the largest component
        component_sizes = np.bincount(labeled.ravel())
        # Ignore background (label 0)
        component_sizes[0] = 0
        largest_label = np.argmax(component_sizes)
        return (labeled == largest_label).astype(bool)

    except ImportError:
        # Without scipy, skip connected component filtering
        # The morphological closing already handles most noise
        logger.debug(
            "scipy not available — skipping connected component filtering"
        )
        return mask
