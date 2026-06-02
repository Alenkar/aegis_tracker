from __future__ import annotations
from typing import List, Dict
import torch
from ..candidates.types import Candidate
from ..utils.box_ops import iou_xywh, center_distance
from ..config import AegisConfig


def label_candidates(candidates: List[Candidate], gt_bbox, cfg: AegisConfig) -> List[Dict]:
    rows = []
    for c in candidates:
        iou = iou_xywh(c.bbox, gt_bbox)
        cd = center_distance(c.bbox, gt_bbox)
        target = float(iou >= cfg.candidate_positive_iou or cd <= cfg.tiny_center_positive_px)
        hard_neg = float(c.local_score > 0.35 and target == 0.0)
        rows.append({'candidate': c, 'target': target, 'hard_neg': hard_neg, 'iou': iou, 'center_dist': cd})
    return rows


def candidate_to_feature(c: Candidate, cfg: AegisConfig):
    vals = [
        c.local_score, c.objectness, c.quality, c.center_score, c.size_score,
        c.identity_score, c.motion_score, c.final_score,
    ]
    return torch.tensor(vals, dtype=torch.float32)
