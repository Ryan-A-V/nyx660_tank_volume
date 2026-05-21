from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from abc import ABC, abstractmethod


@dataclass
class DepthFrame:
    depth_m: np.ndarray
    timestamp_utc: str
    rgb: Optional[np.ndarray] = None


class CameraBackend(ABC):
    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def get_frame(self) -> DepthFrame:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def is_open(self) -> bool: ...
