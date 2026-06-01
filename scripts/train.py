import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aegistrack.config import config_to_dict, enabled_dataset_entries, load_aegis_config
from aegistrack.tracker import AegisTrackOne
from aegistrack.training.dataset import SOTPairDataset
from aegistrack.training.losses import local_core_loss
from aegistrack.utils.clearml_logger import ClearMLLogger, resolve_clearml_dataset
try:
    from eval_sot import evaluate_sot
except ModuleNotFoundError:
    from scripts.eval_sot import evaluate_sot


def create_tensorboard_writer(cfg):
    if not cfg.tensorboard.enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f'WARNING: TensorBoard is disabled: {exc}', file=sys.stderr)
        return None
    log_dir = Path(cfg.tensorboard.log_dir) if cfg.tensorboard.log_dir else Path(cfg.train.out) / 'tensorboard'
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f'TensorBoard log dir: {log_dir}', flush=True)
    return SummaryWriter(log_dir=str(log_dir))


def report_tensorboard(writer, metrics: dict, iteration: int):
    if writer is None:
        return
    for name, value in metrics.items():
        if value is not None:
            writer.add_scalar(name, float(value), int(iteration))
    writer.flush()


def dataset_split_path(root: str, split: str) -> str:
    split_path = Path(root) / split
    return str(split_path) if split_path.exists() else root


def configured_or_split(root: str, configured: str, split: str) -> str:
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            relative_to_root = Path(root) / configured_path
            if relative_to_root.exists():
                return str(relative_to_root)
        return configured
    return dataset_split_path(root, split)


def save_checkpoint(tracker: AegisTrackOne, cfg, path: Path, epoch: int | None = None):
    torch.save({'cfg': config_to_dict(cfg), 'epoch': epoch, 'local_core': tracker.local_core.state_dict()}, path)


def float_tensor_dict(values: dict) -> dict:
    return {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value for key, value in values.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()

    cfg = load_aegis_config(args.config)
    train_cfg = cfg.train
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if not cfg.clearml.task_name:
        cfg.clearml.task_name = 'train'

    clearml = ClearMLLogger(
        enabled=cfg.clearml.enabled,
        project_name=cfg.clearml.project_name,
        task_name=cfg.clearml.task_name,
        queue_name=cfg.clearml.queue_name,
        remote=cfg.clearml.remote,
        args={'config_path': args.config, 'config_values': config_to_dict(cfg)},
    )
    data = resolve_clearml_dataset(cfg.clearml.dataset_id, train_cfg.data)
    if data:
        train_data = dataset_split_path(data, 'train')
        val_data = configured_or_split(data, cfg.validation.data, 'val')
    else:
        train_data = enabled_dataset_entries(cfg, 'train')
        val_data = cfg.validation.data or enabled_dataset_entries(cfg, 'val')
        if not train_data:
            ap.error('train.data is required or at least one datasets.sources entry must be enabled')

    out_dir = Path(train_cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = create_tensorboard_writer(cfg)
    checkpoint_path = out_dir / 'aegis_last.pt'
    best_auc_path = out_dir / 'aegis_best_auc.pt'
    best_candidate_path = out_dir / 'aegis_best_candidate_recall_5.pt'
    best_p20_path = out_dir / 'aegis_best_p20.pt'

    print(f'Train data: {train_data}', flush=True)
    if cfg.validation.enabled:
        print(
            f'Validation data: {val_data} '
            f'(every={cfg.validation.every}, '
            f'max_sequences={cfg.validation.max_sequences or "all"}, '
            f'max_frames_per_sequence={cfg.validation.max_frames_per_sequence or "all"})',
            flush=True,
        )

    ds = SOTPairDataset(train_data, cfg, train_cfg.pairs_per_epoch, max_gap=train_cfg.max_gap, augment=True)
    dl = DataLoader(
        ds,
        batch_size=train_cfg.batch,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=cfg.device.startswith('cuda'),
        timeout=int(getattr(train_cfg, 'dataloader_timeout', 0)) if train_cfg.num_workers > 0 else 0,
    )
    tracker = AegisTrackOne(cfg)
    if train_cfg.ckpt:
        tracker.load(train_cfg.ckpt, strict=False)

    params = list(tracker.local_core.parameters())
    opt = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=0.05)
    use_amp = cfg.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_auc = None
    best_candidate_recall_5 = None
    best_p20 = None

    for ep in range(train_cfg.epochs):
        tracker.local_core.train()
        pbar = tqdm(dl, desc=f'train {ep + 1}/{train_cfg.epochs}')
        sums = {k: 0.0 for k in [
            'loss_total','loss_task','loss_center','loss_box','loss_giou','loss_l1','loss_offset','loss_logwh',
            'loss_objectness','loss_quality','wr','hr','pred_w_mean','pred_h_mean','gt_w_mean','gt_h_mean'
        ]}
        n_batches = 0
        for batch in pbar:
            template = batch['template'].to(cfg.device, non_blocking=True)
            search = batch['search'].to(cfg.device, non_blocking=True)
            gt = batch['gt_bbox'].to(cfg.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = tracker.local_core.forward_train(template, search)
            with torch.amp.autocast('cuda', enabled=False):
                losses = local_core_loss(float_tensor_dict(pred), gt.float(), cfg)
                total_loss = losses['loss']
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()

            sums['loss_total'] += float(total_loss.item())
            sums['loss_task'] += float(losses['task'].item())
            sums['loss_center'] += float(losses['center'].item())
            sums['loss_box'] += float(losses['box'].item())
            sums['loss_giou'] += float(losses['giou'].item())
            sums['loss_l1'] += float(losses['l1'].item())
            sums['loss_offset'] += float(losses['offset'].item())
            sums['loss_logwh'] += float(losses['logwh'].item())
            sums['loss_objectness'] += float(losses['obj'].item())
            sums['loss_quality'] += float(losses['quality'].item())
            for key in ['wr','hr','pred_w_mean','pred_h_mean','gt_w_mean','gt_h_mean']:
                sums[key] += float(losses[key].item())
            n_batches += 1
            pbar.set_postfix(loss=float(total_loss.item()), box=float(losses['box'].item()), task=float(losses['task'].item()), wr=float(losses['wr'].item()), hr=float(losses['hr'].item()))

        if n_batches:
            metrics = {f'train/{k}': v / n_batches for k, v in sums.items()}
            metrics['train/lr'] = float(opt.param_groups[0]['lr'])
            metrics['train/grad_norm'] = float(grad_norm)
            clearml.report_many(metrics, ep + 1)
            report_tensorboard(tb_writer, metrics, ep + 1)
        save_checkpoint(tracker, cfg, checkpoint_path, epoch=ep + 1)
        clearml.upload_artifact('checkpoint_last', str(checkpoint_path))

        if cfg.validation.enabled and (ep + 1) % max(1, cfg.validation.every) == 0:
            tqdm.write(f'validation {ep + 1}/{train_cfg.epochs}: running on {cfg.validation.max_sequences or "all"} sequence(s), {cfg.validation.max_frames_per_sequence or "all"} frame(s)/sequence')
            val_metrics = evaluate_sot(val_data, str(checkpoint_path), cfg.device, verbose=True, cfg=cfg, max_sequences=cfg.validation.max_sequences, max_frames_per_sequence=cfg.validation.max_frames_per_sequence)
            if val_metrics:
                tqdm.write('validation ' + f"auc={val_metrics.get('val/success_auc', 0.0):.4f} p20={val_metrics.get('val/precision_20', 0.0):.4f} iou={val_metrics.get('val/iou_mean', 0.0):.4f} ce={val_metrics.get('val/center_error', 0.0):.2f} cr5={val_metrics.get('val/candidate_recall_5', 0.0):.4f}")
            clearml.report_many(val_metrics, ep + 1)
            report_tensorboard(tb_writer, val_metrics, ep + 1)
            auc = val_metrics.get('val/success_auc')
            if auc is not None and (best_auc is None or auc > best_auc):
                best_auc = auc
                save_checkpoint(tracker, cfg, best_auc_path, epoch=ep + 1)
                clearml.upload_artifact('checkpoint_best_auc', str(best_auc_path))
            p20 = val_metrics.get('val/precision_20')
            if p20 is not None and (best_p20 is None or p20 > best_p20):
                best_p20 = p20
                save_checkpoint(tracker, cfg, best_p20_path, epoch=ep + 1)
                clearml.upload_artifact('checkpoint_best_p20', str(best_p20_path))
            cr5 = val_metrics.get('val/candidate_recall_5')
            if cr5 is not None and (best_candidate_recall_5 is None or cr5 > best_candidate_recall_5):
                best_candidate_recall_5 = cr5
                save_checkpoint(tracker, cfg, best_candidate_path, epoch=ep + 1)
                clearml.upload_artifact('checkpoint_best_candidate_recall_5', str(best_candidate_path))

    clearml.close()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    main()
