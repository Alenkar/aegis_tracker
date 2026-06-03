from __future__ import annotations
from typing import Optional, Tuple
import time
import numpy as np
import torch

from .config import AegisConfig, config_to_dict
from .core.local_core import LocalCore
from .utils.crop_policy import adaptive_search_crop_policy
from .utils.image import crop_with_context
from .utils.box_ops import BBox, clamp_bbox, bbox_center, make_bbox_from_center
from .candidates.types import Candidate, Decision, TrackOutput, TrackState


def _candidate_metric(c: Candidate, name: str, default: float = 0.0) -> float:
    return float(getattr(c, name, default))


def _candidate_center_nms(
    candidates: list[Candidate],
    radius_px: float,
    max_keep: int,
) -> list[Candidate]:
    """Center-distance NMS for tiny objects.

    IoU-NMS is unstable for small boxes: two duplicate peaks shifted by a few
    pixels can have low IoU. We suppress candidates whose centers are too close
    to already kept candidates.
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: float(c.local_score), reverse=True)
    kept: list[Candidate] = []

    for cand in ordered:
        cx, cy = cand.center
        is_duplicate = False
        for prev in kept:
            px, py = prev.center
            if float(np.hypot(cx - px, cy - py)) <= radius_px:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        kept.append(cand)
        if len(kept) >= max_keep:
            break

    return kept


class StableSizeFallback:
    """Stable object size reference used only to damp noisy bbox size predictions."""

    def __init__(self, cfg: AegisConfig):
        self.cfg = cfg
        self.stable_size = (8.0, 8.0)

    def reset(self, bbox: BBox):
        self.stable_size = (max(1.0, float(bbox[2])), max(1.0, float(bbox[3])))

    def update(self, bbox: BBox):
        lr = self.cfg.stable_size_lr_tiny if self.cfg.profile == 'tiny_uav' else self.cfg.stable_size_lr_generic
        sw, sh = self.stable_size
        self.stable_size = ((1.0 - lr) * sw + lr * float(bbox[2]), (1.0 - lr) * sh + lr * float(bbox[3]))


class AegisTrackOne:
    """Single-path UETrack-like tracker.

    Active runtime is LocalCore top-K, center-distance NMS, motion-aware
    selection, TRACKING/LOST state transitions, reinit quarantine, stable size
    fallback, adaptive crop, and debug scores.
    """

    def __init__(self, cfg: Optional[AegisConfig] = None):
        self.cfg = cfg or AegisConfig()
        self.local_core = LocalCore(self.cfg)
        self.stable_box = StableSizeFallback(self.cfg)
        self.state = TrackState.LOST
        self.initialized = False
        self.prev_frame = None
        self.current_bbox: Optional[BBox] = None
        self.last_good_bbox: Optional[BBox] = None
        self.frame_idx = 0
        self.bad_count = 0
        self.good_count = 0
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_checkpoint_meta = {}
        self.pending_recovery_bbox: Optional[BBox] = None
        self.initial_target_token: Optional[torch.Tensor] = None

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
        self.initial_target_token = self.local_core.encode_target(frame, init_bbox).detach()
        self.stable_box.reset(init_bbox)
        self.state = TrackState.TRACKING
        self.current_bbox = init_bbox
        self.last_good_bbox = init_bbox
        self.prev_frame = frame.copy()
        self.frame_idx = 0
        self.bad_count = 0
        self.good_count = 0
        self.velocity[:] = 0.0
        self.pending_recovery_bbox = None
        self.initialized = True


    def terminate(self):
        """Terminate the current track completely.

        This is different from LOST: no bbox is carried forward, no recovery is
        attempted, and callers must explicitly initialize again.
        """
        self.state = TrackState.LOST
        self.initialized = False
        self.prev_frame = None
        self.current_bbox = None
        self.last_good_bbox = None
        self.pending_recovery_bbox = None
        self.initial_target_token = None
        self.bad_count = 0
        self.good_count = 0
        self.velocity[:] = 0.0

    def _adaptive_crop_policy(self, frame: np.ndarray, anchor: Optional[BBox] = None):
        # Use the actual search anchor size. In LOST this can be pending recovery
        # or last_good, not a corrupted current bbox.
        if anchor is None:
            assert self.current_bbox is not None
            anchor = self.current_bbox
        H, W = frame.shape[:2]
        gain = 1.0
        if self.state == TrackState.LOST:
            gain = float(getattr(self.cfg, 'crop_lost_gain', 1.9))
        return adaptive_search_crop_policy(anchor, W, H, self.cfg, gain=gain)

    def _adaptive_crop_side(self, frame: np.ndarray, anchor: Optional[BBox] = None):
        return self._adaptive_crop_policy(frame, anchor).crop_side

    def _select_anchor(self) -> tuple[BBox, str]:
        assert self.current_bbox is not None
        if self.state == TrackState.LOST:
            if self.pending_recovery_bbox is not None:
                return self.pending_recovery_bbox, 'pending'
            if self.last_good_bbox is not None:
                return self.last_good_bbox, 'last_good'
        return self.current_bbox, 'current'

    def _motion_ok(self, pred_bbox: BBox) -> tuple[bool, float, float]:
        # In TRACKING, do not accept jumps to a far false peak.
        # In LOST, large jumps are expected and should be handled by recovery.
        if self.current_bbox is None or self.state == TrackState.LOST:
            return True, 0.0, 1.0
        pcx, pcy = bbox_center(self.current_bbox)
        cx, cy = bbox_center(pred_bbox)
        jump_px = float(np.hypot(cx - pcx, cy - pcy))
        max_side = max(float(self.current_bbox[2]), float(self.current_bbox[3]), 1.0)
        max_jump = float(getattr(self.cfg, 'uetrack_max_tracking_jump_factor', 4.0)) * max_side
        # Allow an absolute floor for fast tiny motion, but do not allow full-frame jumps.
        max_jump = max(max_jump, float(getattr(self.cfg, 'uetrack_max_tracking_jump_min_px', 48.0)))
        return jump_px <= max_jump, jump_px, max_jump

    def _motion_stats_for_candidate(self, cand: Candidate) -> tuple[bool, float, float]:
        """Return candidate motion stats relative to current bbox.

        In LOST, large motion is expected, so the candidate is always motion-ok.
        """
        if self.current_bbox is None or self.state == TrackState.LOST:
            return True, 0.0, 1.0

        pcx, pcy = bbox_center(self.current_bbox)
        cx, cy = cand.center
        jump_px = float(np.hypot(cx - pcx, cy - pcy))
        max_side = max(float(self.current_bbox[2]), float(self.current_bbox[3]), 1.0)
        max_jump = float(getattr(self.cfg, 'uetrack_max_tracking_jump_factor', 4.0)) * max_side
        max_jump = max(max_jump, float(getattr(self.cfg, 'uetrack_max_tracking_jump_min_px', 48.0)))
        return jump_px <= max_jump, jump_px, max_jump

    def _candidate_shape_score(self, cand: Candidate) -> float:
        """Response-map quality score independent of raw peak value."""
        psr = _candidate_metric(cand, 'psr')
        margin = _candidate_metric(cand, 'peak_margin')
        ratio = _candidate_metric(cand, 'peak_ratio', 1.0)

        psr_s = 1.0 / (1.0 + np.exp(-(psr - 4.0) / 1.5))
        margin_s = 1.0 / (1.0 + np.exp(-(margin - 0.05) / 0.03))
        ratio_s = 1.0 / (1.0 + np.exp(-(ratio - 1.20) / 0.15))
        return float(np.clip(0.45 * psr_s + 0.35 * margin_s + 0.20 * ratio_s, 0.0, 1.0))

    def _motion_consistency_score(self, jump_px: float, max_jump_px: float) -> float:
        return float(np.exp(-((float(jump_px) / max(float(max_jump_px), 1e-6)) ** 2)))

    def _size_prior_score(self, bbox: BBox) -> float:
        sw, sh = self.stable_box.stable_size
        w = max(float(bbox[2]), 1e-6)
        h = max(float(bbox[3]), 1e-6)
        log_w = np.log(w / max(float(sw), 1e-6))
        log_h = np.log(h / max(float(sh), 1e-6))
        return float(np.exp(-0.5 * (log_w * log_w + log_h * log_h)))

    def _objectness_quality_score(self, cand: Candidate) -> float:
        return float(np.clip(0.5 * float(cand.objectness) + 0.5 * float(cand.quality), 0.0, 1.0))

    def _identity_score(self, cand: Candidate) -> float:
        if self.initial_target_token is None or cand.visual_emb is None:
            return 0.0
        token = self.initial_target_token.to(cand.visual_emb.device)
        return float(torch.dot(cand.visual_emb, token).clamp(-1.0, 1.0).item())

    def _recovery_verifier_ok(self, cand: Candidate) -> bool:
        shape_ok = float(getattr(cand, 'response_shape_score', 0.0)) >= float(getattr(self.cfg, 'recovery_min_shape_score', 0.45))
        identity_ok = float(getattr(cand, 'identity_score', -1.0)) >= float(getattr(self.cfg, 'recovery_min_identity_score', 0.10))
        size_ok = float(getattr(cand, 'size_prior_score', 0.0)) >= float(getattr(self.cfg, 'recovery_min_size_prior_score', 0.35))
        cand.recovery_shape_ok = bool(shape_ok)
        cand.recovery_identity_ok = bool(identity_ok)
        cand.recovery_size_ok = bool(size_ok)
        cand.recovery_verifier_ok = bool(shape_ok and identity_ok and size_ok)
        return bool(cand.recovery_verifier_ok)

    def _score_candidate(
        self,
        cand: Candidate,
        *,
        motion_ok: bool,
        jump_px: float,
        max_jump_px: float,
    ) -> None:
        response_shape = self._candidate_shape_score(cand)
        motion = self._motion_consistency_score(jump_px, max_jump_px) if motion_ok else 0.0
        size_prior = self._size_prior_score(cand.bbox)
        obj_quality = self._objectness_quality_score(cand)
        identity = self._identity_score(cand)

        tracking_score = (
            float(getattr(self.cfg, 'tracking_score_shape_weight', 0.70)) * response_shape
            + float(getattr(self.cfg, 'tracking_score_motion_weight', 0.30)) * motion
        )
        recovery_score = (
            float(getattr(self.cfg, 'recovery_score_shape_weight', 0.35)) * response_shape
            + float(getattr(self.cfg, 'recovery_score_identity_weight', 0.35)) * max(identity, 0.0)
            + float(getattr(self.cfg, 'recovery_score_size_weight', 0.15)) * size_prior
            + float(getattr(self.cfg, 'recovery_score_objectness_quality_weight', 0.15)) * obj_quality
        )

        cand.response_shape_score = float(np.clip(response_shape, 0.0, 1.0))
        cand.motion_score = float(np.clip(motion, 0.0, 1.0))
        cand.size_prior_score = float(np.clip(size_prior, 0.0, 1.0))
        cand.objectness_quality_score = float(obj_quality)
        cand.identity_score = float(identity)
        cand.tracking_score = float(np.clip(tracking_score, 0.0, 1.0))
        cand.recovery_score = float(np.clip(recovery_score, 0.0, 1.0))
        self._recovery_verifier_ok(cand)

    def _select_candidate_from_topk(
        self,
        candidates: list[Candidate],
    ) -> tuple[Optional[Candidate], str, float, float, float, int, int]:
        """Select one candidate from top-K after center-NMS.

        TRACKING:
            reject far candidates and pick the best motion-consistent candidate.
            This allows taking top-2/top-3 when raw top-1 is a far false peak.

        LOST:
            do not apply motion gate; recovery may be far away.
        """
        if not candidates:
            return None, 'no_candidates', 0.0, 0.0, 1.0, 0, 0

        radius_px = float(getattr(self.cfg, 'center_nms_radius_px', 10.0))
        max_keep = (
            int(getattr(self.cfg, 'center_nms_max_keep_lost', 8))
            if self.state == TrackState.LOST
            else int(getattr(self.cfg, 'center_nms_max_keep_tracking', 5))
        )
        nms_candidates = _candidate_center_nms(candidates, radius_px=radius_px, max_keep=max_keep)

        if not nms_candidates:
            return None, 'nms_empty', 0.0, 0.0, 1.0, len(candidates), 0

        # LOST: choose by recovery score. Large jumps are valid recovery hypotheses.
        if self.state == TrackState.LOST:
            for cand in nms_candidates:
                cand.motion_ok = True
                cand.jump_px = 0.0
                cand.max_jump_px = 1.0
                self._score_candidate(cand, motion_ok=True, jump_px=0.0, max_jump_px=1.0)
            best = max(
                nms_candidates,
                key=lambda c: float(getattr(c, 'recovery_score', 0.0)),
            )
            best.selection_score = float(best.recovery_score)
            return best, 'lost_recovery_score_after_nms', best.selection_score, 0.0, 1.0, len(candidates), len(nms_candidates)

        valid: list[Candidate] = []

        for cand in nms_candidates:
            motion_ok, jump_px, max_jump_px = self._motion_stats_for_candidate(cand)
            cand.motion_ok = bool(motion_ok)
            cand.jump_px = float(jump_px)
            cand.max_jump_px = float(max_jump_px)

            if not motion_ok:
                self._score_candidate(cand, motion_ok=False, jump_px=jump_px, max_jump_px=max_jump_px)
                continue

            self._score_candidate(cand, motion_ok=True, jump_px=jump_px, max_jump_px=max_jump_px)
            cand.selection_score = float(cand.tracking_score)
            valid.append(cand)

        if valid:
            best = max(valid, key=lambda c: float(getattr(c, 'selection_score', c.local_score)))
            return (
                best,
                'tracking_motion_topk',
                float(best.selection_score),
                float(getattr(best, 'jump_px', 0.0)),
                float(getattr(best, 'max_jump_px', 1.0)),
                len(candidates),
                len(nms_candidates),
            )

        raw_top1 = nms_candidates[0]
        return (
            None,
            'all_rejected_by_motion',
            float(raw_top1.local_score),
            float(getattr(raw_top1, 'jump_px', 0.0)),
            float(getattr(raw_top1, 'max_jump_px', 1.0)),
            len(candidates),
            len(nms_candidates),
        )

    def _active_score(self, best: Candidate) -> Tuple[float, bool]:
        if self.state == TrackState.LOST:
            score = float(getattr(best, 'recovery_score', 0.0))
            threshold = float(getattr(self.cfg, 'recovery_score_thr', getattr(self.cfg, 'uetrack_trust_thr', 0.45)))
        else:
            score = float(getattr(best, 'tracking_score', 0.0))
            threshold = float(getattr(self.cfg, 'tracking_score_thr', getattr(self.cfg, 'uetrack_trust_thr', 0.45)))
        return float(np.clip(score, 0.0, 1.0)), bool(score >= threshold)

    def _blend_predicted_size(self, pred_bbox: BBox) -> tuple[BBox, float]:
        key = 'size_pred_blend_tiny' if self.cfg.profile == 'tiny_uav' else 'size_pred_blend_generic'
        blend = float(np.clip(getattr(self.cfg, key, 1.0), 0.0, 1.0))
        if blend >= 1.0:
            return pred_bbox, blend
        sw, sh = self.stable_box.stable_size
        w = blend * float(pred_bbox[2]) + (1.0 - blend) * float(sw)
        h = blend * float(pred_bbox[3]) + (1.0 - blend) * float(sh)
        return make_bbox_from_center(*bbox_center(pred_bbox), w, h), blend

    @torch.no_grad()
    def track(self, frame: np.ndarray) -> TrackOutput:
        if not self.initialized or self.current_bbox is None:
            raise RuntimeError('Call initialize(frame, bbox) first')
        t0 = time.perf_counter()
        H, W = frame.shape[:2]
        self.frame_idx += 1

        # Predict around current bbox in TRACKING, but around pending/last_good in LOST.
        anchor, anchor_source = self._select_anchor()
        crop_policy = self._adaptive_crop_policy(frame, anchor)
        if getattr(self.cfg, 'debug', False):
            print(
                'crop_policy '
                f'crop_side={crop_policy.crop_side:.2f} '
                f'search_factor={crop_policy.search_factor:.2f} '
                f'object_in_search_px={crop_policy.object_in_search_px:.2f}',
                flush=True,
            )
        crop, meta = crop_with_context(frame, anchor, crop_policy.crop_side)
        local_out = self.local_core.forward_local(frame, crop, meta, self.stable_box.stable_size, self.state)
        candidates = local_out.topk_candidates
        best, select_reason, selection_score, selected_jump_px, selected_max_jump_px, raw_topk_count, nms_topk_count = (
            self._select_candidate_from_topk(candidates)
        )
        if best is None:
            bbox = self.current_bbox
            active_score = 0.0
            good = False
            decision = Decision.HOLD_LAST_GOOD
            size_source = 'hold_no_motion_candidate' if candidates else 'hold_no_candidates'
            motion_ok = False
            jump_px = float(selected_jump_px)
            max_jump_px = float(selected_max_jump_px)

            self.bad_count += 1
            if self.state == TrackState.TRACKING and self.bad_count >= int(getattr(self.cfg, 'uetrack_lost_frames', 3)):
                self.state = TrackState.LOST
                decision = Decision.LOST
                self.good_count = 0
                self.pending_recovery_bbox = None
        else:
            raw_pred_bbox = clamp_bbox(best.bbox, W, H)
            pred_bbox, size_blend = self._blend_predicted_size(raw_pred_bbox)
            pred_bbox = clamp_bbox(pred_bbox, W, H)
            active_score, score_good = self._active_score(best)
            motion_ok = bool(getattr(best, 'motion_ok', True))
            jump_px = float(getattr(best, 'jump_px', selected_jump_px))
            max_jump_px = float(getattr(best, 'max_jump_px', selected_max_jump_px))
            good = bool(score_good and (motion_ok or self.state == TrackState.LOST))
            if self.state == TrackState.TRACKING:
                if good:
                    bbox = pred_bbox
                    decision = Decision.ACCEPT_RAW
                    size_source = 'predicted'
                    self.bad_count = 0
                    self.good_count += 1
                    self.pending_recovery_bbox = None
                else:
                    self.bad_count += 1
                    # Do not jump to a far false peak in TRACKING. Hold last bbox until LOST.
                    bbox = self.current_bbox if self.current_bbox is not None else pred_bbox
                    decision = Decision.HOLD_LAST_GOOD
                    size_source = 'hold_motion' if not motion_ok else 'predicted_uncertain'
                    if self.bad_count >= int(getattr(self.cfg, 'uetrack_lost_frames', 3)):
                        self.state = TrackState.LOST
                        decision = Decision.LOST
                        self.good_count = 0
                        self.pending_recovery_bbox = None
            else:
                # In LOST, large motion is allowed. Use candidate as tentative recovery anchor.
                bbox = pred_bbox
                size_source = 'recovered'
                recovery_verified = bool(getattr(best, 'recovery_verifier_ok', False))
                if score_good and recovery_verified:
                    self.pending_recovery_bbox = pred_bbox
                    self.good_count += 1
                    decision = Decision.REINIT_QUARANTINED
                    if self.good_count >= int(getattr(self.cfg, 'uetrack_recover_frames', 2)):
                        self.state = TrackState.TRACKING
                        self.bad_count = 0
                        self.pending_recovery_bbox = None
                else:
                    self.good_count = 0
                    decision = Decision.LOST
                    if not recovery_verified:
                        size_source = 'rejected_recovery_verifier'

        bbox = clamp_bbox(bbox, W, H)
        old_cx, old_cy = bbox_center(self.current_bbox)
        new_cx, new_cy = bbox_center(bbox)
        self.velocity = 0.8 * self.velocity + 0.2 * np.array([new_cx - old_cx, new_cy - old_cy], dtype=np.float32)
        self.current_bbox = bbox
        if good and self.state == TrackState.TRACKING:
            self.last_good_bbox = bbox
            # stable size is a reference only, updated from accepted prediction.
            self.stable_box.update(bbox)
        self.prev_frame = frame.copy()
        elapsed = max(1e-6, time.perf_counter() - t0)
        scores = {
            'match_score': float(best.local_score if best else 0.0),
            'second_score': _candidate_metric(best, 'second_score') if best else 0.0,
            'peak_margin': _candidate_metric(best, 'peak_margin') if best else 0.0,
            'peak_ratio': _candidate_metric(best, 'peak_ratio', 1.0) if best else 1.0,
            'psr': _candidate_metric(best, 'psr') if best else 0.0,
            'tracking_score': float(getattr(best, 'tracking_score', 0.0)) if best else 0.0,
            'recovery_score': float(getattr(best, 'recovery_score', 0.0)) if best else 0.0,
            'active_score': float(active_score),
            'response_shape_score': float(getattr(best, 'response_shape_score', 0.0)) if best else 0.0,
            'motion_score': float(getattr(best, 'motion_score', 0.0)) if best else 0.0,
            'identity_score': float(getattr(best, 'identity_score', 0.0)) if best else 0.0,
            'size_prior_score': float(getattr(best, 'size_prior_score', 0.0)) if best else 0.0,
            'objectness_quality_score': float(getattr(best, 'objectness_quality_score', 0.0)) if best else 0.0,
            'recovery_verifier_ok': float(getattr(best, 'recovery_verifier_ok', False)) if best else 0.0,
            'recovery_shape_ok': float(getattr(best, 'recovery_shape_ok', False)) if best else 0.0,
            'recovery_identity_ok': float(getattr(best, 'recovery_identity_ok', False)) if best else 0.0,
            'recovery_size_ok': float(getattr(best, 'recovery_size_ok', False)) if best else 0.0,
            'recovery_min_shape_score': float(getattr(self.cfg, 'recovery_min_shape_score', 0.45)),
            'recovery_min_identity_score': float(getattr(self.cfg, 'recovery_min_identity_score', 0.10)),
            'recovery_min_size_prior_score': float(getattr(self.cfg, 'recovery_min_size_prior_score', 0.35)),
            'objectness': float(best.objectness) if best else 0.0,
            'quality': float(best.quality) if best else 0.0,
            'bbox_size_source': size_source,
            'pred_w': float(getattr(best, 'predicted_size', (bbox[2], bbox[3]))[0]) if best else bbox[2],
            'pred_h': float(getattr(best, 'predicted_size', (bbox[2], bbox[3]))[1]) if best else bbox[3],
            'size_blend': float(size_blend) if best else 1.0,
            'bbox_decode_center_window': float(getattr(best, 'bbox_decode_center_window', 1)) if best else 1.0,
            'bbox_decode_size_window': float(getattr(best, 'bbox_decode_size_window', 1)) if best else 1.0,
            'stable_w': float(self.stable_box.stable_size[0]),
            'stable_h': float(self.stable_box.stable_size[1]),
            'bad_count': float(self.bad_count),
            'good_count': float(self.good_count),
            'anchor_source': anchor_source,
            'motion_ok': float(motion_ok),
            'jump_px': float(jump_px),
            'max_jump_px': float(max_jump_px),
            'selection_score': float(selection_score),
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
            confidence=float(active_score),
            scores=scores,
            quality={'bbox_size_source': size_source, 'select_reason': select_reason},
            candidates=candidates,
            reason=[size_source, select_reason],
            fps=1.0 / elapsed,
        )
