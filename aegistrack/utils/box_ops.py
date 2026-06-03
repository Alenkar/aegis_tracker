from __future__ import annotations
from typing import Tuple
import math

BBox = Tuple[float, float, float, float]


def xywh_to_xyxy(b: BBox) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return x, y, x + w, y + h


def bbox_center(b: BBox) -> Tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


def bbox_area(b: BBox) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def make_bbox_from_center(cx: float, cy: float, w: float, h: float) -> BBox:
    return cx - w / 2.0, cy - h / 2.0, max(1.0, w), max(1.0, h)


def clamp_bbox(b: BBox, W: int, H: int) -> BBox:
    x, y, w, h = b
    w = max(1.0, min(float(w), float(W)))
    h = max(1.0, min(float(h), float(H)))
    x = max(0.0, min(float(x), float(W) - w))
    y = max(0.0, min(float(y), float(H) - h))
    return x, y, w, h


def iou_xywh(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter + 1e-9
    return float(inter / union)


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(math.hypot(ax - bx, ay - by))
