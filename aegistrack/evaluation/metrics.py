from __future__ import annotations
from typing import List
import numpy as np
from ..utils.box_ops import BBox, iou_xywh, center_distance
from ..candidates.types import Candidate


def success_auc(preds: List[BBox], gts: List[BBox]):
    ious = np.array([iou_xywh(p, g) for p, g in zip(preds, gts)], dtype=np.float32)
    thrs = np.linspace(0, 1, 101)
    return float(np.mean([(ious >= t).mean() for t in thrs]))


def precision_at(preds: List[BBox], gts: List[BBox], thr: float = 20.0):
    ds = np.array([center_distance(p, g) for p, g in zip(preds, gts)], dtype=np.float32)
    return float((ds <= thr).mean())


def candidate_recall_at_k(candidates: List[Candidate], gt: BBox, k: int = 5, iou_thr: float = 0.3, center_thr: float = 4.0):
    ordered = sorted(candidates, key=lambda c: c.final_score if c.final_score != 0 else c.local_score, reverse=True)[:k]
    return float(any(iou_xywh(c.bbox, gt) >= iou_thr or center_distance(c.bbox, gt) <= center_thr for c in ordered))


def bbox_explosion(pred: BBox, stable_size, high: float = 3.0):
    area = pred[2] * pred[3]
    st = stable_size[0] * stable_size[1] + 1e-9
    return float(area / st > high)
