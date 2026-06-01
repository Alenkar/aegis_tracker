from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
from ..config import AegisConfig
from ..candidates.types import Candidate, TrackState
from ..temporal.state import TemporalState
from ..utils.box_ops import bbox_area


class DecisionNet(nn.Module):
    """Trainable target-instance decision head.

    Outputs: accept, present, update, recover, lost, reject.
    It is trained as a ranker over top-K local hypotheses, not as a class detector.
    """

    def __init__(self, in_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.out = nn.Linear(hidden, 6)

    def forward(self, x):
        return self.out(self.net(x))


class PresenceNet(nn.Module):
    """Trainable presence/absence/recovery head."""

    def __init__(self, in_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 6), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)  # present, absent_crop, out, occluded, takeover, need_recovery


def build_candidate_features(cfg: AegisConfig, c: Candidate, temporal: TemporalState, state: TrackState):
    """Runtime/train feature layout shared by DecisionNet and PresenceNet.

    This function intentionally does not use GT labels. It only contains values available
    during inference. That keeps the learned ranker aligned with full Aegis runtime.
    """
    src_hash = (hash(c.source.value) % 17) / 17.0
    vals = [
        float(c.local_score),
        float(c.objectness),
        float(c.quality),
        float(c.center_score),
        float(c.size_score),
        float(c.identity_score),
        float(c.negative_score),
        float(c.distractor_score),
        float(c.motion_score),
        float(c.presence_score),
        float(c.recovery_score),
        float(c.update_score),
        float(c.uncertainty),
        float(temporal.drift_risk),
        float(temporal.switch_risk),
        float(temporal.lost_risk),
        float(temporal.update_risk),
        float(temporal.recovery_risk),
        src_hash,
        float(state == TrackState.TRACKING),
        float(state == TrackState.UNCERTAIN),
        float(state == TrackState.VERIFYING),
        float(state == TrackState.LOST),
        float(state == TrackState.REACQUIRED),
    ]
    vals += [0.0] * (cfg.candidate_feature_dim - len(vals))
    return torch.tensor(vals[:cfg.candidate_feature_dim], dtype=torch.float32, device=cfg.device)


class RuntimeDecisionHead:
    """Inference wrapper.

    The deterministic path is not a heuristic replacement for Aegis; it is the
    non-learned target-instance selector used until the learned head is enabled.
    It combines raw localization with identity, motion, size and negative memory.
    """

    def __init__(self, cfg: AegisConfig, net: DecisionNet | None = None):
        self.cfg = cfg
        self.net = net

    def build_features(self, c: Candidate, temporal: TemporalState, state: TrackState):
        return build_candidate_features(self.cfg, c, temporal, state)

    @torch.no_grad()
    def score(self, c: Candidate, temporal: TemporalState, state: TrackState):
        if self.net is not None:
            x = self.build_features(c, temporal, state).unsqueeze(0)
            logits = self.net(x)[0]
            probs = torch.sigmoid(logits)
            learned_accept = float(probs[0].item())
            # Blend learned accept with deterministic safety score. This prevents an early weak
            # learned head from completely overriding the local tracker.
            det = self._deterministic_score(c, temporal, state)
            c.final_score = float(np.clip(0.55 * det + 0.45 * learned_accept, 0.0, 1.0))
            c.presence_score = max(c.presence_score, float(probs[1].item()))
            c.update_score = float(probs[2].item())
            c.recovery_score = float(probs[3].item())
            return c
        c.final_score = self._deterministic_score(c, temporal, state)
        c.update_score = float(np.clip(c.final_score - temporal.update_risk, 0.0, 1.0))
        c.recovery_score = float(np.clip(0.45 * c.identity_score + 0.25 * c.motion_score + 0.30 * c.local_score - 0.35 * c.negative_score, 0.0, 1.0))
        return c

    def _deterministic_score(self, c: Candidate, temporal: TemporalState, state: TrackState):
        raw = float(np.clip(c.local_score, 0.0, 1.0))
        obj = float(np.clip(c.objectness, 0.0, 1.0))
        qual = float(np.clip(c.quality, 0.0, 1.0))
        motion = float(np.clip(c.motion_score, 0.0, 1.0))
        ident = float(np.clip(c.identity_score, 0.0, 1.0))
        size = float(np.clip(c.size_score, 0.0, 1.0))
        neg = float(np.clip(c.negative_score, 0.0, 1.0))
        dist = float(np.clip(c.distractor_score, 0.0, 1.0))
        uncertainty = float(np.clip(c.uncertainty, 0.0, 1.0))
        if state == TrackState.LOST:
            score = 0.38 * raw + 0.18 * obj + 0.16 * qual + 0.20 * ident + 0.08 * size - 0.25 * neg - 0.20 * dist - 0.05 * uncertainty
        else:
            score = (
                self.cfg.selector_raw_weight * raw
                + self.cfg.selector_motion_weight * motion
                + self.cfg.selector_identity_weight * ident
                + self.cfg.selector_size_weight * size
                + self.cfg.selector_quality_weight * (0.5 * obj + 0.5 * qual)
                - self.cfg.selector_negative_weight * neg
                - self.cfg.selector_distractor_weight * dist
                - 0.08 * temporal.drift_risk
                - 0.05 * uncertainty
            )
        return float(np.clip(score, 0.0, 1.0))


class RuntimePresenceHead:
    def __init__(self, cfg: AegisConfig, net: PresenceNet | None = None):
        self.cfg = cfg
        self.net = net
        self.low_present_count = 0

    def reset(self):
        self.low_present_count = 0

    @torch.no_grad()
    def forward(self, candidates, temporal: TemporalState, state: TrackState, feature_builder=None):
        if self.net is not None and candidates:
            best = max(candidates, key=lambda c: c.final_score)
            x = feature_builder(best, temporal, state) if feature_builder is not None else build_candidate_features(self.cfg, best, temporal, state)
            probs = self.net(x.unsqueeze(0))[0]
            present = float(probs[0].item())
            if present < 0.35:
                self.low_present_count += 1
            else:
                self.low_present_count = 0
            return {
                'present': present,
                'absent_crop': max(float(probs[1].item()), float(self.low_present_count >= self.cfg.low_present_for_lost_frames)),
                'out': float(probs[2].item()),
                'occluded': float(probs[3].item()),
                'takeover': float(probs[4].item()),
                'need_recovery': float(probs[5].item()),
            }
        if candidates:
            best = max(candidates, key=lambda c: c.final_score)
            present = float(np.clip(0.50 * best.local_score + 0.25 * best.identity_score + 0.20 * best.motion_score + 0.05 * best.size_score, 0, 1))
        else:
            present = 0.0
        if present < 0.30:
            self.low_present_count += 1
        else:
            self.low_present_count = 0
        return {
            'present': present,
            'absent_crop': float(self.low_present_count >= self.cfg.low_present_for_lost_frames),
            'out': float(present < 0.12 and temporal.lost_risk > 0.8),
            'occluded': float(0.15 <= present < 0.40),
            'takeover': temporal.switch_risk,
            'need_recovery': float(state in (TrackState.LOST, TrackState.UNCERTAIN, TrackState.VERIFYING) or self.low_present_count >= 1),
        }

    def _build_features(self, c: Candidate, temporal: TemporalState, state: TrackState):
        return build_candidate_features(self.cfg, c, temporal, state)
