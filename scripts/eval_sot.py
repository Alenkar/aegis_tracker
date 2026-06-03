import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tqdm import tqdm
from aegistrack import AegisTrackOne
from aegistrack.config import config_to_dict, enabled_dataset_entries, load_aegis_config
from aegistrack.candidates.types import Decision, TrackState
from aegistrack.utils.box_ops import center_distance, iou_xywh
from aegistrack.utils.image import crop_with_context
from aegistrack.training.dataset import find_sequence_dirs, load_sequences, read_frame
from aegistrack.evaluation.metrics import bbox_explosion, candidate_recall_at_k, success_auc, precision_at
from aegistrack.utils.clearml_logger import ClearMLLogger, resolve_clearml_dataset


def add_clearml_args(ap: argparse.ArgumentParser, clearml_cfg, default_task_name: str):
    ap.add_argument('--clearml', action='store_true', default=clearml_cfg.enabled)
    ap.add_argument('--clearml-project', default=clearml_cfg.project_name)
    ap.add_argument('--clearml-task-name', default=clearml_cfg.task_name or default_task_name)
    ap.add_argument('--clearml-queue', default=clearml_cfg.queue_name)
    ap.add_argument('--clearml-remote', action='store_true', default=clearml_cfg.remote)
    ap.add_argument('--clearml-dataset-id', default=clearml_cfg.dataset_id)


def first_local_core_checksum(tracker: AegisTrackOne) -> float:
    first_param = next(tracker.local_core.parameters())
    return float(first_param.detach().float().sum().item())


def print_checkpoint_debug(ckpt_path: str, tracker: AegisTrackOne):
    meta = getattr(tracker, 'last_checkpoint_meta', {})
    print(f'Loaded checkpoint: {Path(ckpt_path).resolve()}', flush=True)
    print(f'Loaded checkpoint epoch: {meta.get("epoch")}', flush=True)
    print(f'local_core checksum: {first_local_core_checksum(tracker):.6f}', flush=True)


@torch.no_grad()
def raw_local_candidates(tracker: AegisTrackOne, frame):
    if not tracker.initialized or tracker.current_bbox is None:
        return []
    anchor, _ = tracker._anchor()
    crop_policy = tracker._crop_policy(frame, anchor)
    crop, meta = crop_with_context(frame, anchor, crop_policy.crop_side)
    stable_size = (tracker.current_bbox[2], tracker.current_bbox[3])
    local_out = tracker.local_core.forward_local(frame, crop, meta, stable_size, tracker.state)
    return local_out.topk_candidates


def evaluate_sot(
    data: str | list,
    ckpt: str = '',
    device: str = 'cuda',
    verbose: bool = True,
    cfg=None,
    max_sequences: int = 0,
    max_frames_per_sequence: int = 0,
):
    if cfg is None:
        cfg = load_aegis_config(device=device)
    roots = data if isinstance(data, list) else [{'path': data, 'modalities': []}]
    aucs, precs = [], []
    center_errors, ious = [], []
    candidate_recall_1 = 0.0
    candidate_recall_5 = 0.0
    raw_candidate_recall_1 = 0.0
    raw_candidate_recall_5 = 0.0
    candidate_frames = 0
    raw_candidate_frames = 0
    target_switch_count = 0
    bbox_explosion_count = 0
    bad_state_commit_count = 0
    recovery_attempts = 0
    recovery_success = 0
    false_reinit_count = 0
    time_to_recover = []
    lost_frames = 0
    tracked_frames = 0

    evaluated = 0
    checkpoint_printed = False
    for item in roots:
        root = Path(item['path'] if isinstance(item, dict) else item)
        modalities = tuple(item.get('modalities') or ()) if isinstance(item, dict) else ()
        if not root.exists():
            if verbose:
                print(f'Validation data path does not exist: {root}')
            continue
        for seq in find_sequence_dirs(root):
            if max_sequences and evaluated >= max_sequences:
                break
            loaded_sequences = load_sequences(seq, modalities)
            for loaded in loaded_sequences:
                if max_sequences and evaluated >= max_sequences:
                    break
                evaluated += 1
                imgs, gts, seq_name = loaded
                tracker = AegisTrackOne(cfg)
                if ckpt:
                    tracker.load(ckpt, strict=False)
                    if not checkpoint_printed:
                        print_checkpoint_debug(ckpt, tracker)
                        checkpoint_printed = True
                frame = read_frame(imgs[0])
                if frame is None:
                    continue
                tracker.initialize(frame, gts[0])
                preds = [gts[0]]
                eval_gts = [gts[0]]
                ious.append(iou_xywh(gts[0], gts[0]))
                center_errors.append(center_distance(gts[0], gts[0]))
                stable_size = (gts[0][2], gts[0][3])
                lost_start = None
                frame_items = list(enumerate(imgs[1:], start=1))
                if max_frames_per_sequence:
                    frame_items = frame_items[:max_frames_per_sequence]
                for frame_idx, p in tqdm(frame_items, desc=seq_name, disable=not verbose):
                    frame = read_frame(p)
                    if frame is None:
                        continue
                    gt = gts[frame_idx]
                    raw_candidates = raw_local_candidates(tracker, frame)
                    raw_candidate_recall_1 += candidate_recall_at_k(raw_candidates, gt, k=1)
                    raw_candidate_recall_5 += candidate_recall_at_k(raw_candidates, gt, k=5)
                    raw_candidate_frames += 1
                    out = tracker.track(frame)
                    preds.append(out.target_bbox)
                    eval_gts.append(gt)
                    iou = iou_xywh(out.target_bbox, gt)
                    ious.append(iou)
                    center_errors.append(center_distance(out.target_bbox, gt))
                    candidate_recall_1 += candidate_recall_at_k(out.candidates, gt, k=1)
                    candidate_recall_5 += candidate_recall_at_k(out.candidates, gt, k=5)
                    candidate_frames += 1
                    target_switch_count += int(out.scores.get('switch_risk', 0.0) > 0.5)
                    bbox_explosion_count += int(bbox_explosion(out.target_bbox, stable_size) > 0.0)
                    bad_state_commit_count += int(out.decision == Decision.ACCEPT_RAW and iou < 0.3)
                    recovery_attempts += int(out.state == TrackState.TRACKING and out.decision == Decision.ACCEPT_RAW and lost_start is not None)
                    recovery_success += int(out.state == TrackState.TRACKING and out.decision == Decision.ACCEPT_RAW and lost_start is not None and iou >= 0.3)
                    false_reinit_count += int(out.state == TrackState.TRACKING and out.decision == Decision.ACCEPT_RAW and lost_start is not None and iou < 0.3)
                    tracked_frames += 1
                    if out.state == TrackState.LOST:
                        lost_frames += 1
                        if lost_start is None:
                            lost_start = frame_idx
                    elif lost_start is not None and iou >= 0.3:
                        time_to_recover.append(frame_idx - lost_start)
                        lost_start = None
                aucs.append(success_auc(preds, eval_gts)); precs.append(precision_at(preds, eval_gts, 20))
                if verbose:
                    print(seq_name, 'AUC', aucs[-1], 'P20', precs[-1])
        if max_sequences and evaluated >= max_sequences:
            break

    if not aucs:
        return {}
    metrics = {
        'val/success_auc': sum(aucs) / len(aucs),
        'val/precision_20': sum(precs) / len(precs),
        'val/center_error': sum(center_errors) / len(center_errors),
        'val/iou_mean': sum(ious) / len(ious),
        'val/candidate_recall_1': candidate_recall_1 / candidate_frames if candidate_frames else 0.0,
        'val/candidate_recall_5': candidate_recall_5 / candidate_frames if candidate_frames else 0.0,
        'drift/target_switch_count': float(target_switch_count),
        'drift/bbox_explosion_count': float(bbox_explosion_count),
        'drift/bad_state_commit_count': float(bad_state_commit_count),
        'recovery/success_rate': recovery_success / recovery_attempts if recovery_attempts else 0.0,
        'recovery/false_reinit_count': float(false_reinit_count),
        'recovery/time_to_recover_mean': sum(time_to_recover) / len(time_to_recover) if time_to_recover else 0.0,
        'state/lost_ratio': lost_frames / tracked_frames if tracked_frames else 0.0,
    }
    if verbose:
        print('MEAN AUC', metrics['val/success_auc'], 'MEAN P20', metrics['val/precision_20'])
        print(
            'raw_candidate_recall_1',
            raw_candidate_recall_1 / raw_candidate_frames if raw_candidate_frames else 0.0,
            'raw_candidate_recall_5',
            raw_candidate_recall_5 / raw_candidate_frames if raw_candidate_frames else 0.0,
        )
    return metrics


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='')
    pre_args, _ = pre.parse_known_args()
    cfg_defaults = load_aegis_config(pre_args.config)

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument('--data', default='')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--device', default=cfg_defaults.device)
    ap.add_argument('--max-sequences', type=int, default=cfg_defaults.validation.max_sequences)
    ap.add_argument('--max-frames-per-sequence', type=int, default=cfg_defaults.validation.max_frames_per_sequence)
    add_clearml_args(ap, cfg_defaults.clearml, 'eval_sot')
    args = ap.parse_args()

    cfg = load_aegis_config(args.config, device=args.device)
    cfg.clearml.enabled = args.clearml
    cfg.clearml.project_name = args.clearml_project
    cfg.clearml.task_name = args.clearml_task_name
    cfg.clearml.queue_name = args.clearml_queue
    cfg.clearml.remote = args.clearml_remote
    cfg.clearml.dataset_id = args.clearml_dataset_id
    clearml = ClearMLLogger(
        enabled=cfg.clearml.enabled,
        project_name=cfg.clearml.project_name,
        task_name=cfg.clearml.task_name,
        queue_name=cfg.clearml.queue_name,
        remote=cfg.clearml.remote,
        args={**vars(args), 'config_values': config_to_dict(cfg)},
    )
    args.data = resolve_clearml_dataset(cfg.clearml.dataset_id, args.data)
    eval_data = args.data or enabled_dataset_entries(cfg, 'val')
    if not eval_data:
        ap.error('--data is required or at least one datasets.sources entry must be enabled')
    metrics = evaluate_sot(
        eval_data,
        args.ckpt,
        args.device,
        verbose=True,
        cfg=cfg,
        max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    if metrics:
        clearml.report_many(metrics, iteration=0)
    clearml.close()

if __name__ == '__main__':
    main()
