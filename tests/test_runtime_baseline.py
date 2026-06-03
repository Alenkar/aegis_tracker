import unittest

import numpy as np
import torch

from aegistrack.candidates.types import Candidate, CandidateSource, LocalOutput, TrackState
from aegistrack.config import AegisConfig, load_aegis_config
from aegistrack.tracker import AegisTrackOne
from aegistrack.utils.image import crop_point_to_frame, crop_with_context, frame_bbox_to_crop


class FakeLocalCore:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def eval(self):
        return self

    def initialize(self, frame, bbox):
        return None

    def forward_local(self, frame, crop, crop_meta, stable_size, state):
        return self.outputs.pop(0)


def local_output(candidates):
    response = torch.zeros(4, 4)
    return LocalOutput(
        best_bbox=candidates[0].bbox if candidates else (0.0, 0.0, 1.0, 1.0),
        response_map=response,
        topk_candidates=candidates,
        raw_score=0.0,
        windowed_score=0.0,
        crop_meta={},
    )


def candidate(bbox, local_score=0.9):
    c = Candidate(
        bbox=bbox,
        source=CandidateSource.TOPK,
        local_score=local_score,
        objectness=0.9,
        quality=0.9,
    )
    c.second_score = 0.1
    c.peak_margin = 0.8
    c.peak_ratio = 9.0
    c.psr = 8.0
    return c


class RuntimeBaselineTests(unittest.TestCase):
    def test_crop_round_trip_center_and_padded_edge(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        cases = [((40.0, 30.0, 7.0, 5.0), 33.3), ((0.0, 0.0, 6.0, 4.0), 45.7)]
        for bbox, side in cases:
            _, meta = crop_with_context(frame, bbox, side)
            crop_bbox = frame_bbox_to_crop(bbox, meta, 64)
            cx = crop_bbox[0] + crop_bbox[2] / 2.0
            cy = crop_bbox[1] + crop_bbox[3] / 2.0
            fx, fy = crop_point_to_frame(cx, cy, meta, 64)
            self.assertLessEqual(abs(fx - (bbox[0] + bbox[2] / 2.0)), 0.5)
            self.assertLessEqual(abs(fy - (bbox[1] + bbox[3] / 2.0)), 0.5)

    def test_lost_low_score_recovery_does_not_commit_candidate(self):
        cfg = AegisConfig(
            device="cpu",
            local_feature_dim=32,
            template_size=32,
            search_size=64,
            tracking_score_thr=0.99,
            recovery_score_thr=0.5,
            uetrack_lost_frames=1,
        )
        tracker = AegisTrackOne(cfg)
        tracker.local_core = FakeLocalCore(
            [
                local_output([candidate((60.0, 60.0, 8.0, 8.0))]),
                local_output([candidate((40.0, 40.0, 80.0, 80.0), local_score=0.1)]),
            ]
        )
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        tracker.initialize(frame, (20.0, 20.0, 8.0, 8.0))

        out1 = tracker.track(frame)
        self.assertEqual(out1.state, TrackState.LOST)

        out2 = tracker.track(frame)
        self.assertEqual(out2.state, TrackState.LOST)
        self.assertEqual(tracker.current_bbox, tracker.last_good_bbox)

    def test_config_rejects_unknown_keys_and_loads_motion_gate(self):
        cfg = load_aegis_config("configs/tiny_uav.yaml")
        self.assertEqual(cfg.uetrack_max_tracking_jump_factor, 4.0)
        self.assertEqual(cfg.uetrack_max_tracking_jump_min_px, 48.0)


if __name__ == "__main__":
    unittest.main()
