from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CalibrateRequest(BaseModel):
    frames: Optional[int] = None


class MeasureResponse(BaseModel):
    timestamp_utc: str
    estimated_volume_m3: float
    estimated_volume_liters: Optional[float]
    relative_fill_ratio: Optional[float]
    occupied_surface_area_m2: float
    average_fill_height_m: float
    max_fill_height_m: float
    valid_pixel_ratio: float
    notes: list[str]
