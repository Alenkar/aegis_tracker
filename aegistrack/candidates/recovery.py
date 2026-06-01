from __future__ import annotations
from typing import Callable, List, Tuple
import cv2
import numpy as np
from ..config import AegisConfig
from .types import Candidate, CandidateSource, BBox
from ..utils.box_ops import make_bbox_from_center, clamp_bbox
from .pool import nms_candidates

DetectorFn = Callable[[np.ndarray], List[Tuple[BBox, float, str]]]


class RecoveryProposer:
    def __init__(self, cfg: AegisConfig):
        self.cfg = cfg

    def log_dog(self, frame, stable_size) -> List[Candidate]:
        if not self.cfg.enable_log_dog_recovery:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        g = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (3, 3), 0)
        responses = []
        for sigma in self.cfg.log_sigmas:
            k = int(max(3, 2 * round(3 * sigma) + 1))
            blur = cv2.GaussianBlur(g, (k, k), sigma)
            lap = cv2.Laplacian(blur, cv2.CV_32F)
            resp = (sigma ** 2) * np.abs(lap)
            thr = float(resp.mean() + 1.5 * resp.std())
            ys, xs = np.where(resp > thr)
            if len(xs) > 500:
                idx = np.argsort(resp[ys, xs])[-500:]
                xs, ys = xs[idx], ys[idx]
            for x, y in zip(xs, ys):
                responses.append((float(resp[y, x]), int(x), int(y)))
        responses.sort(reverse=True, key=lambda v: v[0])
        sw, sh = stable_size
        out = [Candidate(make_bbox_from_center(float(x), float(y), sw, sh), CandidateSource.LOG_DOG, local_score=min(1.0, s * 10), objectness=min(1.0, s * 10), quality=0.25) for s, x, y in responses[:self.cfg.max_log_candidates]]
        return nms_candidates(out, 0.4, self.cfg.max_log_candidates)

    def tiles(self, frame, stable_size) -> List[Candidate]:
        if not self.cfg.enable_tile_recovery:
            return []
        H, W = frame.shape[:2]
        gx, gy = self.cfg.tile_grid
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        out = []
        sw, sh = stable_size
        for iy in range(gy):
            for ix in range(gx):
                x1, y1 = int(ix * W / gx), int(iy * H / gy)
                x2, y2 = int((ix + 1) * W / gx), int((iy + 1) * H / gy)
                tile = gray[y1:y2, x1:x2]
                if tile.size == 0: continue
                blur = cv2.GaussianBlur(tile, (0, 0), 3.0)
                sal = cv2.absdiff(tile, blur)
                _, mx, _, loc = cv2.minMaxLoc(sal)
                cx, cy = x1 + loc[0], y1 + loc[1]
                score = float(mx) / 255.0
                out.append(Candidate(make_bbox_from_center(cx, cy, sw, sh), CandidateSource.TILE, local_score=score, objectness=score, quality=0.2))
        return nms_candidates(out, 0.4, self.cfg.max_tile_candidates)


class DetectorProposalAdapter:
    def __init__(self, cfg: AegisConfig, detector_fn: DetectorFn | None = None):
        self.cfg = cfg
        self.detector_fn = detector_fn

    def propose(self, frame, allowed: bool) -> List[Candidate]:
        if not allowed or not self.cfg.enable_detector_proposals or self.detector_fn is None:
            return []
        H, W = frame.shape[:2]
        out = []
        for bbox, score, cls in self.detector_fn(frame)[:self.cfg.max_detector_candidates]:
            out.append(Candidate(clamp_bbox(bbox, W, H), CandidateSource.DETECTOR, local_score=float(score), objectness=float(score), quality=0.25, reason=[f'detector_class={cls}']))
        return out
