from __future__ import annotations
import cv2
import numpy as np
from ..utils.box_ops import BBox, clamp_bbox, xyxy_to_xywh


class EgoMotionLite:
    def __init__(self, enabled=True):
        self.enabled = enabled

    @staticmethod
    def gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def estimate(self, prev_frame, frame):
        if not self.enabled or prev_frame is None:
            return np.eye(2, 3, dtype=np.float32)
        g0, g1 = self.gray(prev_frame), self.gray(frame)
        pts0 = cv2.goodFeaturesToTrack(g0, maxCorners=400, qualityLevel=0.01, minDistance=8)
        if pts0 is None or len(pts0) < 12:
            return np.eye(2, 3, dtype=np.float32)
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None)
        if pts1 is None or st is None:
            return np.eye(2, 3, dtype=np.float32)
        src, dst = pts0[st.squeeze() == 1], pts1[st.squeeze() == 1]
        if len(src) < 12:
            return np.eye(2, 3, dtype=np.float32)
        A, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        return A.astype(np.float32) if A is not None else np.eye(2, 3, dtype=np.float32)

    @staticmethod
    def warp_bbox(b: BBox, A, W: int, H: int) -> BBox:
        x, y, w, h = b
        pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)
        pts_h = np.concatenate([pts, np.ones((4, 1), dtype=np.float32)], axis=1)
        out = (A @ pts_h.T).T
        x1, y1 = out[:, 0].min(), out[:, 1].min()
        x2, y2 = out[:, 0].max(), out[:, 1].max()
        return clamp_bbox(xyxy_to_xywh((x1, y1, x2, y2)), W, H)
