from __future__ import annotations
from typing import List
from .types import Candidate
from ..utils.box_ops import iou_xywh


class CandidatePool:
    def __init__(self, max_size: int = 96):
        self.max_size = max_size
        self.items: List[Candidate] = []

    def add(self, c: Candidate):
        self.items.append(c)

    def extend(self, cs):
        self.items.extend(cs)

    def nms(self, iou_thr: float = 0.65):
        ordered = sorted(self.items, key=lambda c: c.final_score if c.final_score != 0 else c.local_score, reverse=True)
        keep = []
        for c in ordered:
            if all(iou_xywh(c.bbox, k.bbox) < iou_thr for k in keep):
                keep.append(c)
            if len(keep) >= self.max_size:
                break
        self.items = keep
        return self.items


def nms_candidates(candidates: List[Candidate], iou_thr: float = 0.65, max_keep: int = 96):
    pool = CandidatePool(max_keep)
    pool.extend(candidates)
    return pool.nms(iou_thr)
