from __future__ import annotations
from typing import Dict
import cv2
import numpy as np
import torch
from .box_ops import BBox, bbox_center

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def crop_with_context(frame: np.ndarray, bbox_or_center, side: float):
    H, W = frame.shape[:2]
    if len(bbox_or_center) == 4:
        cx, cy = bbox_center(bbox_or_center)
    else:
        cx, cy = bbox_or_center
    side = float(max(2.0, side))
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - W), max(0, y2 - H)
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    crop = frame[y1c:y2c, x1c:x2c]
    if any(v > 0 for v in (pad_l, pad_t, pad_r, pad_b)):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    meta = {"x1": float(x1), "y1": float(y1), "side": float(side), "frame_w": float(W), "frame_h": float(H)}
    return crop, meta


def crop_to_tensor(crop: np.ndarray, size: int, device: str):
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)


def crop_point_to_frame(px: float, py: float, meta: Dict[str, float], input_size: int):
    scale = meta["side"] / float(input_size)
    return meta["x1"] + px * scale, meta["y1"] + py * scale


def frame_bbox_to_crop(b: BBox, meta: Dict[str, float], input_size: int) -> BBox:
    x, y, w, h = b
    scale = float(input_size) / meta["side"]
    return (x - meta["x1"]) * scale, (y - meta["y1"]) * scale, w * scale, h * scale
