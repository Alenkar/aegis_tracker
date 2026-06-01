import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from aegistrack import AegisTrackOne
from aegistrack.config import load_aegis_config


def parse_bbox(s):
    vals = [float(x) for x in s.replace(',', ' ').split()]
    if len(vals) != 4:
        raise ValueError('--bbox must be x,y,w,h')
    return tuple(vals)


def select_bbox(frame):
    box = cv2.selectROI('select init bbox', frame, False, False)
    cv2.destroyWindow('select init bbox')
    return tuple(float(x) for x in box)


def fmt_bbox(b):
    return ','.join(f'{v:.1f}' for v in b)


def draw_output(frame, bbox, text, candidates=None):
    vis = frame.copy()
    if candidates:
        for c in candidates[:10]:
            bx, by, bw, bh = [int(round(v)) for v in c.bbox]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
            cx, cy = int(round(bx + bw / 2)), int(round(by + bh / 2))
            cv2.drawMarker(vis, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 8, 1)
    if bbox is not None:
        x, y, w, h = [int(round(v)) for v in bbox]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(vis, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def cuda_sync(device: str):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--video', required=True)
    ap.add_argument('--bbox', default='', help='x,y,w,h on first frame')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='test.mp4')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--draw-candidates', action='store_true')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--log-every', type=int, default=1)
    ap.add_argument('--warmup-frames', type=int, default=10)
    ap.add_argument('--csv-log', default='')
    ap.add_argument('--delete-score-thr', type=float, default=0.5,
                    help='Terminate current track completely when confidence is below this threshold. After deletion, tracking stops until manual reinit with R.')
    args = ap.parse_args()

    cfg = load_aegis_config(args.config, device=args.device)
    cfg.egomotion_enabled = False
    cfg.use_learned_runtime_heads = False
    tracker = AegisTrackOne(cfg).to(args.device)
    tracker.load(args.ckpt, strict=False)

    cap = cv2.VideoCapture(args.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25

    cap.set(cv2.CAP_PROP_POS_FRAMES, fps_in * 75)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('Cannot read video')
    bbox = parse_bbox(args.bbox) if args.bbox else select_bbox(frame)
    tracker.initialize(frame, bbox)
    print(f'frame=0 decision=init bbox={fmt_bbox(bbox)}', flush=True)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (W, H)) if args.out else None
    csv_f = open(args.csv_log, 'w', newline='', encoding='utf-8') if args.csv_log else None
    csv_writer = None
    if csv_f is not None:
        fields = ['frame','state','decision','bbox_x','bbox_y','bbox_w','bbox_h','confidence','match_score','second_score','peak_margin','peak_ratio','psr','bbox_size_source','pred_w','pred_h','stable_w','stable_h','size_blend','bbox_decode_center_window','bbox_decode_size_window','bad_count','good_count','anchor_source','motion_ok','jump_px','max_jump_px','selection_score','raw_topk_count','nms_topk_count','track_deleted','tracking_active','time_ms','fps']
        csv_writer = csv.DictWriter(csv_f, fieldnames=fields)
        csv_writer.writeheader()

    if wr is not None:
        wr.write(draw_output(frame, bbox, 'INIT'))
    if not args.headless:
        cv2.imshow('AegisTrack', draw_output(frame, bbox, 'INIT'))

    times_ms = []
    frame_idx = 1
    tracking_active = True
    ok, frame = cap.read()
    while ok:
        if not tracking_active:
            vis = draw_output(frame, None, f'TRACK DELETED - press R to reinit', None)
            if wr is not None:
                wr.write(vis)
            if not args.headless:
                cv2.imshow('AegisTrack', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                if key == ord('r'):
                    bbox = select_bbox(frame)
                    tracker.initialize(frame, bbox)
                    tracking_active = True
                    print(f'frame={frame_idx} decision=reinit bbox={fmt_bbox(bbox)}', flush=True)
            frame_idx += 1
            ok, frame = cap.read()
            continue

        cuda_sync(args.device)
        t0 = time.perf_counter()
        out = tracker.track(frame)
        cuda_sync(args.device)
        time_ms = (time.perf_counter() - t0) * 1000.0
        inst_fps = 1000.0 / max(time_ms, 1e-6)
        bbox = out.target_bbox
        scores = out.scores
        track_deleted = bool(out.confidence < args.delete_score_thr)
        if track_deleted:
            tracker.terminate()
            tracking_active = False
        draw_bbox = None if track_deleted else bbox
        if frame_idx > args.warmup_frames:
            times_ms.append(time_ms)
        if args.log_every and frame_idx % args.log_every == 0:
            print(
                f"frame={frame_idx} state={out.state.value} decision={out.decision.value} "
                f"deleted={int(track_deleted)} active={int(tracking_active)} conf={out.confidence:.3f} match={scores.get('match_score',0):.3f} "
                f"psr={scores.get('psr',0):.2f} margin={scores.get('peak_margin',0):.3f} "
                f"bbox={fmt_bbox(bbox)} pred=({scores.get('pred_w',0):.1f},{scores.get('pred_h',0):.1f}) "
                f"stable=({scores.get('stable_w',0):.1f},{scores.get('stable_h',0):.1f}) "
                f"anchor={scores.get('anchor_source','')} source={scores.get('bbox_size_source','')} "
                f"motion_ok={scores.get('motion_ok',0):.0f} jump={scores.get('jump_px',0):.1f}/{scores.get('max_jump_px',0):.1f} "
                f"sel={scores.get('selection_score',0):.3f} topk={int(scores.get('raw_topk_count',0))}/{int(scores.get('nms_topk_count',0))} "
                f"time_ms={time_ms:.2f} fps={inst_fps:.1f}",
                flush=True,
            )
        if csv_writer is not None:
            csv_writer.writerow({
                'frame': frame_idx,
                'state': out.state.value,
                'decision': out.decision.value,
                'bbox_x': bbox[0], 'bbox_y': bbox[1], 'bbox_w': bbox[2], 'bbox_h': bbox[3],
                'confidence': out.confidence,
                'match_score': scores.get('match_score', 0.0),
                'second_score': scores.get('second_score', 0.0),
                'peak_margin': scores.get('peak_margin', 0.0),
                'peak_ratio': scores.get('peak_ratio', 1.0),
                'psr': scores.get('psr', 0.0),
                'bbox_size_source': scores.get('bbox_size_source', ''),
                'pred_w': scores.get('pred_w', 0.0), 'pred_h': scores.get('pred_h', 0.0),
                'stable_w': scores.get('stable_w', 0.0), 'stable_h': scores.get('stable_h', 0.0),
                'size_blend': scores.get('size_blend', 1.0),
                'bbox_decode_center_window': scores.get('bbox_decode_center_window', 1.0),
                'bbox_decode_size_window': scores.get('bbox_decode_size_window', 1.0),
                'bad_count': scores.get('bad_count', 0.0), 'good_count': scores.get('good_count', 0.0),
                'anchor_source': scores.get('anchor_source', ''),
                'motion_ok': scores.get('motion_ok', 0.0),
                'jump_px': scores.get('jump_px', 0.0),
                'max_jump_px': scores.get('max_jump_px', 0.0),
                'selection_score': scores.get('selection_score', 0.0),
                'raw_topk_count': scores.get('raw_topk_count', 0.0),
                'nms_topk_count': scores.get('nms_topk_count', 0.0),
                'track_deleted': int(track_deleted),
                'tracking_active': int(tracking_active),
                'time_ms': time_ms, 'fps': inst_fps,
            })
        vis_text = f"{out.state.value} {out.confidence:.2f} {scores.get('bbox_size_source','')}"
        if track_deleted:
            vis_text = f"DELETED score<{args.delete_score_thr:.2f} {out.confidence:.2f}"
        vis = draw_output(frame, draw_bbox, vis_text, out.candidates if (args.draw_candidates and not track_deleted) else None)
        if wr is not None:
            wr.write(vis)
        if not args.headless:
            cv2.imshow('AegisTrack', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if key == ord('r'):
                bbox = select_bbox(frame)
                tracker.initialize(frame, bbox)
                print(f'frame={frame_idx} decision=reinit bbox={fmt_bbox(bbox)}', flush=True)
        frame_idx += 1
        ok, frame = cap.read()

    cap.release()
    if wr is not None:
        wr.release()
    if csv_f is not None:
        csv_f.close()
    if not args.headless:
        cv2.destroyAllWindows()
    if times_ms:
        arr = np.asarray(times_ms, dtype=np.float64)
        fps_arr = 1000.0 / np.maximum(arr, 1e-6)
        print(f'timing tracked_frames={frame_idx-1} warmup_frames={args.warmup_frames} measured_frames={len(arr)} time_ms_mean={arr.mean():.3f} time_ms_std={arr.std():.3f} fps_mean={fps_arr.mean():.3f} fps_std={fps_arr.std():.3f}', flush=True)


if __name__ == '__main__':
    main()
