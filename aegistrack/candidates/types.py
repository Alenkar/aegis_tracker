from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch

BBox = Tuple[float, float, float, float]


class TrackState(str, Enum):
    TRACKING = "TRACKING"
    LOST = "LOST"


class CandidateSource(str, Enum):
    LOCAL = "local"
    TOPK = "topk"


class Decision(str, Enum):
    ACCEPT_RAW = "ACCEPT_RAW"
    HOLD_LAST_GOOD = "HOLD_LAST_GOOD"
    LOST = "LOST"


@dataclass
class Candidate:
    bbox: BBox
    source: CandidateSource
    local_score: float = 0.0
    objectness: float = 0.0
    quality: float = 0.0
    center_score: float = 0.0
    size_score: float = 0.0
    motion_score: float = 0.0
    final_score: float = 0.0
    scalar_features: Optional[torch.Tensor] = None
    reason: List[str] = field(default_factory=list)

    @property
    def center(self):
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    @property
    def size(self):
        return self.bbox[2], self.bbox[3]

    @property
    def area(self):
        return max(0.0, self.bbox[2]) * max(0.0, self.bbox[3])


@dataclass
class LocalOutput:
    best_bbox: BBox
    response_map: torch.Tensor
    topk_candidates: List[Candidate]
    raw_score: float
    windowed_score: float
    crop_meta: Dict[str, float]


@dataclass
class TrackOutput:
    target_bbox: BBox
    state: TrackState
    decision: Decision
    confidence: float
    scores: Dict[str, float]
    quality: Dict[str, float]
    candidates: List[Candidate]
    reason: List[str]
    fps: float
