from __future__ import annotations

from typing import Optional
import time

import numpy as np
import torch

from .config import AegisConfig, config_to_dict
from .core.local_core import LocalCore
from .utils.crop_policy import adaptive_search_crop_policy
from .utils.image import crop_with_context
from .utils.box_ops import BBox, clamp_bbox, bbox_center
from .candidates.types import Candidate, Decision, TrackOutput, TrackState


def _candidate_metric(c: Candidate | None, name: str, default: float = 0.0) -> float:
    if c is None:
        return float(default)
    return float(getattr(c, name, default))


def _candidate_center_nms(candidates: list[Candidate], radius_px: float, max_keep: int) -> list[Candidate]:
    if not candidates:
        return []
    kept: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: float(c.local_score), reverse=True):
        cx, cy = cand.center
        if any(float(np.hypot(cx - px, cy - py)) <= radius_px for px, py in (prev.center for prev in kept)):
            continue
        kept.append(cand)
        if len(kept) >= max_keep:
            break
    return kept


class AegisTrackOne:
    """Minimal runtime tracker.

    Runtime policy is intentionally simple:
    LocalCore top-K -> center NMS -> motion gate in TRACKING -> score threshold.
    No temporal memory, recovery verifier, distractor bookkeeping, or composite scores.
    """

    def __init__(self, cfg: Optional[AegisConfig] = None):
        self.cfg = cfg or AegisConfig()
        self.local_core = LocalCore(self.cfg)
        self.state = TrackState.LOST
        self.initialized = False
        self.current_bbox: Optional[BBox] = None
        self.last_good_bbox: Optional[BBox] = None
        self.frame_idx = 0
        self.bad_count = 0
        self.good_count = 0
        self.last_checkpoint_meta = {}

    def to(self, device: str):
        self.cfg.device = device
        self.local_core.to(device)
        return self

    def save(self, path: str):
        torch.save({'cfg': config_to_dict(self.cfg), 'local_core': self.local_core.state_dict()}, path)

    def load(self, path: str, strict: bool = False):
        ckpt = torch.load(path, map_location=self.cfg.device)
        if 'local_core' in ckpt:
            self.local_core.load_state_dict(ckpt['local_core'], strict=strict)
        elif 'model' in ckpt:
            self.local_core.load_state_dict(ckpt['model'], strict=strict)
        else:
            self.local_core.load_state_dict(ckpt, strict=strict)
        self.last_checkpoint_meta = {'path': path, 'epoch': ckpt.get('epoch') if isinstance(ckpt, dict) else None}
        return self

    def initialize(self, frame: np.ndarray, init_bbox: BBox):
        H, W = frame.shape[:2]
        init_bbox = clamp_bbox(init_bbox, W, H)
        self.local_core.eval()
        self.local_core.initialize(frame, init_bbox)
        self.state = TrackState.TRACKING
        self.current_bbox = init_bbox
        self.last_good_bbox = init_bbox
        self.frame_idx = 0
        self.bad_count = 0
        self.good_count = 0
        self.initialized = True

    def terminate(self):
        self.state = TrackState.LOST
        self.initialized = False
        self.current_bbox = None
        self.last_good_bbox = None
        self.bad_count = 0
        self.good_count = 0

    def _anchor(self) -> tuple[BBox, str]:
        assert self.current_bbox is not None
        if self.state == TrackState.LOST and self.last_good_bbox is not None:
            return self.last_good_bbox, 'last_good'
        return self.current_bbox, 'current'

    def _crop_policy(self, frame: np.ndarray, anchor: BBox):
        H, W = frame.shape[:2]
        gain = float(getattr(self.cfg, 'crop_lost_gain', 1.9)) if self.state == TrackState.LOST else 1.0
        return adaptive_search_crop_policy(anchor, W, H, self.cfg, gain=gain)

    def _motion_stats(self, cand: Candidate) -> tuple[bool, float, float]:
        if self.current_bbox is None or self.state == TrackState.LOST:
            return True, 0.0, 1.0
        pcx, pcy = bbox_center(self.current_bbox)
        cx, cy = cand.center
        jump_px = float(np.hypot(cx - pcx, cy - pcy))
        max_side = max(float(self.current_bbox[2]), float(self.current_bbox[3]), 1.0)
        max_jump = float(getattr(self.cfg, 'uetrack_max_tracking_jump_factor', 4.0)) * max_side
        max_jump = max(max_jump, float(getattr(self.cfg, 'uetrack_max_tracking_jump_min_px', 48.0)))
        return jump_px <= max_jump, jump_px, max_jump

    def _select_candidate(self, candidates: list[Candidate]) -> tuple[Optional[Candidate], str, float, float, int, int]:
        radius = float(getattr(self.cfg, 'center_nms_radius_px', 10.0))
        max_keep = (
            int(getattr(self.cfg, 'center_nms_max_keep_lost', 8))
            if self.state == TrackState.LOST
            else int(getattr(self.cfg, 'center_nms_max_keep_tracking', 5))
        )
        nms = _candidate_center_nms(candidates, radius, max_keep)
        if not nms:
            return None, 'no_candidates', 0.0, 1.0, len(candidates), 0

        if self.state == TrackState.LOST:
            return nms[0], 'lost_top1', 0.0, 1.0, len(candidates), len(nms)

        first_jump, first_max = 0.0, 1.0
        for i, cand in enumerate(nms):
            motion_ok, jump_px, max_jump_px = self._motion_stats(cand)
            cand.motion_ok = bool(motion_ok)
            cand.jump_px = float(jump_px)
            cand.max_jump_px = float(max_jump_px)
            if i == 0:
                first_jump, first_max = jump_px, max_jump_px
            if motion_ok:
                return cand, 'tracking_motion_topk', jump_px, max_jump_px, len(candidates), len(nms)
        return None, 'all_rejected_by_motion', first_jump, first_max, len(candidates), len(nms)

    def _threshold(self) -> float:
        if self.state == TrackState.LOST:
            return float(getattr(self.cfg, 'recovery_score_thr', 0.45))
        return float(getattr(self.cfg, 'tracking_score_thr', 0.45))

    @torch.no_grad()
    def track(self, frame: np.ndarray) -> TrackOutput:
        if not self.initialized or self.current_bbox is None:
            raise RuntimeError('Call initialize(frame, bbox) first')

        t0 = time.perf_counter()
        H, W = frame.shape[:2]
        self.frame_idx += 1

        anchor, anchor_source = self._anchor()
        crop_policy = self._crop_policy(frame, anchor)
        crop, meta = crop_with_context(frame, anchor, crop_policy.crop_side)
        stable_size = (float(self.current_bbox[2]), float(self.current_bbox[3]))
        local_out = self.local_core.forward_local(frame, crop, meta, stable_size, self.state)
        candidates = local_out.topk_candidates
        best, select_reason, jump_px, max_jump_px, raw_topk_count, nms_topk_count = self._select_candidate(candidates)

        bbox = self.current_bbox
        active_score = float(best.local_score) if best is not None else 0.0
        score_good = bool(best is not None and active_score >= self._threshold())
        motion_ok = bool(getattr(best, 'motion_ok', True)) if best is not None else False
        decision = Decision.HOLD_LAST_GOOD
        size_source = 'hold'

        if self.state == TrackState.TRACKING:
            if best is not None and score_good and motion_ok:
                bbox = clamp_bbox(best.bbox, W, H)
                self.last_good_bbox = bbox
                self.bad_count = 0
                self.good_count += 1
                decision = Decision.ACCEPT_RAW
                size_source = 'predicted'
            else:
                self.bad_count += 1
                self.good_count = 0
                if self.bad_count >= int(getattr(self.cfg, 'uetrack_lost_frames', 3)):
                    self.state = TrackState.LOST
                    decision = Decision.LOST
                    size_source = 'lost'
        else:
            if best is not None and score_good:
                bbox = clamp_bbox(best.bbox, W, H)
                self.current_bbox = bbox
                self.last_good_bbox = bbox
                self.state = TrackState.TRACKING
                self.bad_count = 0
                self.good_count = 1
                decision = Decision.ACCEPT_RAW
                size_source = 'recovered'
            else:
                bbox = self.last_good_bbox if self.last_good_bbox is not None else self.current_bbox
                self.bad_count += 1
                self.good_count = 0
                decision = Decision.LOST
                size_source = 'lost'

        bbox = clamp_bbox(bbox, W, H)
        self.current_bbox = bbox
        elapsed = max(1e-6, time.perf_counter() - t0)

        scores = {
            'match_score': active_score,
            'active_score': active_score,
            'tracking_score': active_score if self.state == TrackState.TRACKING else 0.0,
            'recovery_score': active_score if self.state == TrackState.LOST else 0.0,
            'second_score': _candidate_metric(best, 'second_score'),
            'peak_margin': _candidate_metric(best, 'peak_margin'),
            'peak_ratio': _candidate_metric(best, 'peak_ratio', 1.0),
            'psr': _candidate_metric(best, 'psr'),
            'motion_score': 1.0 if motion_ok else 0.0,
            'objectness': float(best.objectness) if best else 0.0,
            'quality': float(best.quality) if best else 0.0,
            'bbox_size_source': size_source,
            'pred_w': float(getattr(best, 'predicted_size', (bbox[2], bbox[3]))[0]) if best else bbox[2],
            'pred_h': float(getattr(best, 'predicted_size', (bbox[2], bbox[3]))[1]) if best else bbox[3],
            'stable_w': float(bbox[2]),
            'stable_h': float(bbox[3]),
            'bad_count': float(self.bad_count),
            'good_count': float(self.good_count),
            'anchor_source': anchor_source,
            'motion_ok': float(motion_ok),
            'jump_px': float(jump_px),
            'max_jump_px': float(max_jump_px),
            'selection_score': active_score,
            'raw_topk_count': float(raw_topk_count),
            'nms_topk_count': float(nms_topk_count),
            'crop_side': float(crop_policy.crop_side),
            'search_factor': float(crop_policy.search_factor),
            'object_in_search_px': float(crop_policy.object_in_search_px),
        }
        return TrackOutput(
            target_bbox=bbox,
            state=self.state,
            decision=decision,
            confidence=active_score,
            scores=scores,
            quality={'bbox_size_source': size_source, 'select_reason': select_reason},
            candidates=candidates,
            reason=[size_source, select_reason],
            fps=1.0 / elapsed,
        )
