from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .box_ops import BBox


@dataclass(frozen=True)
class CropPolicyResult:
    crop_side: float
    search_factor: float
    object_in_search_px: float
    max_obj: float


def adaptive_search_crop_policy(
    bbox: BBox,
    frame_w: int,
    frame_h: int,
    cfg: Any,
    *,
    gain: float = 1.0,
) -> CropPolicyResult:
    max_obj = max(float(bbox[2]), float(bbox[3]), 1.0)
    max_factor = float(getattr(cfg, 'base_search_factor', 18.0))
    min_factor = float(getattr(cfg, 'adaptive_search_min_factor', 2.5))
    scale_ref = max(float(getattr(cfg, 'adaptive_search_scale_ref', 32.0)), 1e-6)

    factor = max_factor / math.sqrt(1.0 + max_obj / scale_ref)
    factor = float(np.clip(factor, min_factor, max_factor))
    crop_side = factor * max_obj * max(float(gain), 1e-6)

    frame_cap = 1.25 * max(float(frame_w), float(frame_h))
    cfg_cap = float(getattr(cfg, 'max_crop_side', frame_cap))
    max_crop_side = min(cfg_cap, frame_cap)
    min_crop_side = min(float(getattr(cfg, 'min_crop_side', 144.0)), max_crop_side)
    crop_side = float(np.clip(crop_side, min_crop_side, max_crop_side))

    search_factor = crop_side / max_obj
    search_size = float(getattr(cfg, 'search_size', 256))
    object_in_search_px = max_obj * search_size / max(crop_side, 1e-6)

    return CropPolicyResult(
        crop_side=crop_side,
        search_factor=float(search_factor),
        object_in_search_px=float(object_in_search_px),
        max_obj=float(max_obj),
    )
