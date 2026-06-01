from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
from ..config import AegisConfig
from ..candidates.types import Candidate, Decision
from ..utils.box_ops import BBox, bbox_center


class TemporalState:
    def __init__(self, cfg: AegisConfig):
        self.cfg = cfg
        self.history: List[Dict[str, float]] = []
        self.velocity = np.zeros(2, dtype=np.float32)
        self.residual_motion = 0.0
        self.drift_risk = 0.0
        self.switch_risk = 0.0
        self.lost_risk = 0.0
        self.update_risk = 0.0
        self.recovery_risk = 0.0

    def reset(self, bbox: BBox):
        self.history.clear()
        self.velocity[:] = 0.0
        self.residual_motion = 0.0
        self.drift_risk = self.switch_risk = self.lost_risk = self.update_risk = self.recovery_risk = 0.0
        cx, cy = bbox_center(bbox)
        self.history.append({'cx': cx, 'cy': cy, 'w': bbox[2], 'h': bbox[3], 'score': 1.0})

    def motion_score(self, c: Candidate, warped_bbox: BBox):
        cx, cy = c.center
        wx, wy = bbox_center(warped_bbox)
        expected = np.array([wx, wy], dtype=np.float32) + self.velocity
        dist = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - expected))
        sigma = self.cfg.motion_sigma0 + self.cfg.motion_sigma_vel_gain * float(np.linalg.norm(self.velocity)) + self.cfg.motion_sigma_unc_gain * self.drift_risk
        return float(np.exp(-(dist * dist) / (2 * sigma * sigma + 1e-9)))

    def as_tensor(self, device: str):
        rows = []
        for h in self.history[-self.cfg.temporal_history:]:
            rows.append([
                h.get('cx', 0), h.get('cy', 0), h.get('w', 0), h.get('h', 0), h.get('score', 0),
                h.get('identity', 0), h.get('negative', 0), h.get('motion', 0), h.get('present', 0), h.get('mem', 0),
                h.get('decision_accept', 0), h.get('decision_lost', 0), self.drift_risk, self.switch_risk, self.lost_risk, self.update_risk,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            ])
        while len(rows) < self.cfg.temporal_history:
            rows.insert(0, [0.0] * self.cfg.temporal_input_dim)
        return torch.tensor(rows, dtype=torch.float32, device=device).unsqueeze(0)

    def update(self, bbox: BBox, candidates: List[Candidate], decision: Decision, presence: Dict[str, float], memory_update: bool):
        cx, cy = bbox_center(bbox)
        if self.history:
            px, py = self.history[-1]['cx'], self.history[-1]['cy']
            v = np.array([cx - px, cy - py], dtype=np.float32)
            self.velocity = 0.8 * self.velocity + 0.2 * v
            self.residual_motion = float(np.linalg.norm(v))
        best = max(candidates, key=lambda c: c.final_score, default=None)
        ambiguity = self._ambiguity(candidates)
        best_score = best.final_score if best else 0.0
        self.drift_risk = float(np.clip(0.55 * self.drift_risk + 0.45 * (1.0 - best_score + ambiguity), 0, 1))
        self.switch_risk = float(np.clip(max([c.distractor_score for c in candidates], default=0.0), 0, 1))
        self.lost_risk = float(np.clip(1.0 - presence.get('present', 0.0), 0, 1))
        self.update_risk = float(np.clip(self.drift_risk + self.switch_risk - 0.3, 0, 1))
        self.history.append({
            'cx': cx, 'cy': cy, 'w': bbox[2], 'h': bbox[3], 'score': best_score,
            'identity': best.identity_score if best else 0.0,
            'negative': best.negative_score if best else 0.0,
            'motion': best.motion_score if best else 0.0,
            'present': presence.get('present', 0.0), 'mem': float(memory_update),
            'decision_accept': float(decision.value.startswith('ACCEPT')),
            'decision_lost': float(decision.value == 'LOST'),
        })
        self.history = self.history[-self.cfg.temporal_history:]

    @staticmethod
    def _ambiguity(cands: List[Candidate]):
        if len(cands) < 2: return 0.0
        s = sorted([c.final_score for c in cands], reverse=True)
        return float(max(0.0, 0.25 - (s[0] - s[1])) / 0.25)
