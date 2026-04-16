from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def depth_to_png_bytes(depth_m: np.ndarray, min_depth: float | None = None, max_depth: float | None = None) -> bytes:
    arr = depth_m.astype(np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    if not np.any(valid):
        arr8 = np.zeros(arr.shape, dtype=np.uint8)
    else:
        mn = float(np.nanmin(arr[valid])) if min_depth is None else float(min_depth)
        mx = float(np.nanmax(arr[valid])) if max_depth is None else float(max_depth)
        if mx <= mn:
            mx = mn + 1e-6
        norm = np.clip((arr - mn) / (mx - mn), 0.0, 1.0)
        arr8 = (norm * 255.0).astype(np.uint8)
    image = Image.fromarray(arr8, mode="L")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
