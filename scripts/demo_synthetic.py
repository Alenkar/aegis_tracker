import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from aegistrack import AegisConfig, AegisTrackOne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    cfg = AegisConfig(device=args.device, depth=1, embed_dim=64, num_heads=4, search_size=128, template_size=64, patch_size=16, topk_tracking=2, topk_lost=2)  # fast smoke-test config
    tracker = AegisTrackOne(cfg)
    frame0 = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(frame0, (320, 180), 3, (255,255,255), -1)
    tracker.initialize(frame0, (316, 177, 8, 6))
    for i in range(1, 2):
        frame = np.zeros_like(frame0)
        cv2.circle(frame, (320 + i*4, 180 + i*2), 3, (255,255,255), -1)
        out = tracker.track(frame)
        print(i, out.state.value, out.decision.value, tuple(round(v, 2) for v in out.target_bbox), round(out.confidence, 3), out.reason)

if __name__ == '__main__':
    main()
