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
        update = float(target == 1.0 and c.negative_score < 0.4 and c.distractor_score < 0.4 and c.uncertainty < 0.4)
        recover = float(target == 1.0 and c.source.value in ('log_dog','tile','detector','shift','motion'))
        rows.append({'candidate': c, 'target': target, 'hard_neg': hard_neg, 'update': update, 'recover': recover, 'iou': iou, 'center_dist': cd})
    return rows


def candidate_to_feature(c: Candidate, cfg: AegisConfig):
    vals = [
        c.local_score, c.objectness, c.quality, c.center_score, c.size_score,
        c.identity_score, c.negative_score, c.distractor_score, c.motion_score,
        c.presence_score, c.recovery_score, c.update_score, c.uncertainty,
    ]
    vals += [0.0] * (cfg.candidate_feature_dim - len(vals))
    return torch.tensor(vals[:cfg.candidate_feature_dim], dtype=torch.float32)
