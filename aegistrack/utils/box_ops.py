from __future__ import annotations
from typing import Tuple, List
import math
import numpy as np

BBox = Tuple[float, float, float, float]


def xywh_to_xyxy(b: BBox) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return x, y, x + w, y + h


def xyxy_to_xywh(b: Tuple[float, float, float, float]) -> BBox:
    x1, y1, x2, y2 = b
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


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


def nms_xywh(boxes: List[BBox], scores: List[float], iou_thr: float, max_keep: int) -> List[int]:
    if not boxes:
        return []
    idxs = list(np.argsort(scores)[::-1])
    keep = []
    while idxs and len(keep) < max_keep:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou_xywh(boxes[i], boxes[j]) < iou_thr]
    return keep


def giou_loss_xyxy(pred, target):
    import torch
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    ix1, iy1 = torch.maximum(px1, tx1), torch.maximum(py1, ty1)
    ix2, iy2 = torch.minimum(px2, tx2), torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    pa = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    ta = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = pa + ta - inter + 1e-7
    iou = inter / union
    cx1, cy1 = torch.minimum(px1, tx1), torch.minimum(py1, ty1)
    cx2, cy2 = torch.maximum(px2, tx2), torch.maximum(py2, ty2)
    ca = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-7
    giou = iou - (ca - union) / ca
    return 1.0 - giou
