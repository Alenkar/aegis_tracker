from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from ..config import AegisConfig
from ..candidates.types import Candidate, CandidateSource, CandidateLife, Decision, TrackState
from ..utils.box_ops import make_bbox_from_center
from ..temporal.state import TemporalState


class StableBoxHead:
    def __init__(self, cfg: AegisConfig):
        self.cfg = cfg
        self.stable_size = (8.0, 8.0)

    def reset(self, bbox):
        self.stable_size = (max(1.0, bbox[2]), max(1.0, bbox[3]))

    def stats(self, c: Candidate):
        sw, sh = self.stable_size
        growth = c.area / (sw * sh + 1e-9)
        aspect = (c.bbox[2] / max(c.bbox[3], 1e-6)) / (sw / max(sh, 1e-6))
        low, high = self.cfg.growth_limits()
        pred_norm_w = c.bbox[2] / max(sw, 1e-6)
        pred_norm_h = c.bbox[3] / max(sh, 1e-6)
        size_bad = growth > high or growth < low or pred_norm_w > (1 + self.cfg.pred_norm_bad) or pred_norm_h > (1 + self.cfg.pred_norm_bad)
        full_bad = c.local_score < self.cfg.min_raw_score and c.objectness < self.cfg.min_objectness and growth > self.cfg.growth_full_bad
        return {
            'growth': float(growth), 'aspect_change': float(aspect),
            'pred_norm_w': float(pred_norm_w), 'pred_norm_h': float(pred_norm_h),
            'size_bad': bool(size_bad), 'full_bad': bool(full_bad),
        }

    def clamp(self, center):
        return make_bbox_from_center(center[0], center[1], self.stable_size[0], self.stable_size[1])

    def update(self, c: Candidate):
        lr = self.cfg.stable_size_lr_tiny if self.cfg.profile == 'tiny_uav' else self.cfg.stable_size_lr_generic
        sw, sh = self.stable_size
        cw, ch = c.size
        self.stable_size = ((1 - lr) * sw + lr * cw, (1 - lr) * sh + lr * ch)


class AntiDriftGate:
    def __init__(self, cfg: AegisConfig, stable: StableBoxHead):
        self.cfg = cfg
        self.stable = stable

    def decide(self, best: Optional[Candidate], candidates: List[Candidate], state: TrackState, presence: Dict[str, float], temporal: TemporalState) -> Tuple[Decision, List[str]]:
        if best is None or not candidates:
            return Decision.LOST, ['no_candidates']
        st = self.stable.stats(best)
        if st['full_bad']:
            return Decision.ROLLBACK, ['full_bad']
        if best.negative_score > self.cfg.max_negative_score:
            best.lifecycle = CandidateLife.REJECTED
            return Decision.VERIFY_MORE, ['negative_memory_match']
        if best.distractor_score > self.cfg.max_distractor_score:
            best.lifecycle = CandidateLife.REJECTED
            return Decision.VERIFY_MORE, ['distractor_match']
        if best.source == CandidateSource.DETECTOR:
            if best.recovery_score > self.cfg.min_recovery_score and best.identity_score > self.cfg.min_identity_score:
                return Decision.REINIT_QUARANTINED, ['detector_candidate_quarantine']
            return Decision.LOST, ['detector_not_identity_confirmed']
        if presence.get('out', 0.0) > 0.5:
            return Decision.LOST, ['presence_out']
        if presence.get('takeover', 0.0) > 0.75:
            return Decision.VERIFY_MORE, ['takeover_risk']
        if best.final_score < self.cfg.min_track_score:
            if state in (TrackState.UNCERTAIN, TrackState.VERIFYING):
                return Decision.LOST, ['low_score_after_uncertain']
            return Decision.ROLLBACK, ['low_score']
        if st['size_bad']:
            if best.local_score > self.cfg.min_raw_score and best.motion_score > 0.15:
                return Decision.ACCEPT_CENTER_CLAMP_SIZE, ['center_good_size_bad']
            return Decision.ROLLBACK, ['size_bad_center_weak']
        if len(candidates) >= 2:
            ordered = sorted(candidates, key=lambda c: c.final_score, reverse=True)
            if ordered[0].final_score - ordered[1].final_score < 0.05 and ordered[1].identity_score > ordered[0].identity_score:
                return Decision.VERIFY_MORE, ['top2_ambiguous']
        if state == TrackState.LOST:
            if best.recovery_score > self.cfg.min_recovery_score:
                return Decision.REINIT_QUARANTINED, ['lost_recovery_quarantine']
            return Decision.LOST, ['lost_no_recovery']
        return Decision.ACCEPT_RAW, ['accepted']
