from __future__ import annotations
import torch
import torch.nn.functional as F

from ..config import AegisConfig


def xywh_to_xyxy_t(b: torch.Tensor) -> torch.Tensor:
    return torch.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], dim=-1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(a[:, 0], b[:, 0])
    y1 = torch.max(a[:, 1], b[:, 1])
    x2 = torch.min(a[:, 2], b[:, 2])
    y2 = torch.min(a[:, 3], b[:, 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    return inter / (area_a + area_b - inter + 1e-6)


def giou_loss_xyxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area_p = (pred[:, 2] - pred[:, 0]).clamp_min(0) * (pred[:, 3] - pred[:, 1]).clamp_min(0)
    area_t = (target[:, 2] - target[:, 0]).clamp_min(0) * (target[:, 3] - target[:, 1]).clamp_min(0)
    union = area_p + area_t - inter + 1e-6
    iou = inter / union
    cx1 = torch.min(pred[:, 0], target[:, 0])
    cy1 = torch.min(pred[:, 1], target[:, 1])
    cx2 = torch.max(pred[:, 2], target[:, 2])
    cy2 = torch.max(pred[:, 3], target[:, 3])
    c_area = (cx2 - cx1).clamp_min(0) * (cy2 - cy1).clamp_min(0) + 1e-6
    giou = iou - (c_area - union) / c_area
    return 1.0 - giou


def make_center_targets(gt_bbox, feat_h: int, feat_w: int, search_size: int, cfg: AegisConfig):
    device = gt_bbox.device
    cx = gt_bbox[:, 0] + gt_bbox[:, 2] / 2
    cy = gt_bbox[:, 1] + gt_bbox[:, 3] / 2
    gx = cx / search_size * feat_w
    gy = cy / search_size * feat_h
    yy, xx = torch.meshgrid(torch.arange(feat_h, device=device), torch.arange(feat_w, device=device), indexing='ij')
    yy = (yy.float() + 0.5)[None]
    xx = (xx.float() + 0.5)[None]
    feature_stride = float(getattr(cfg, 'local_feature_stride', cfg.patch_size))
    sigma = cfg.center_sigma_factor * torch.sqrt(gt_bbox[:, 2].clamp_min(1) * gt_bbox[:, 3].clamp_min(1)) / feature_stride
    sigma = torch.clamp(sigma, min=max(1.0, float(cfg.center_sigma_min)))
    target = torch.exp(-((xx - gx[:, None, None]) ** 2 + (yy - gy[:, None, None]) ** 2) / (2 * sigma[:, None, None] ** 2 + 1e-6))
    return target[:, None]


def focal_bce_logits(logits: torch.Tensor, target: torch.Tensor, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = prob * target + (1 - prob) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - pt).pow(gamma) * ce).mean()


def local_core_loss(out, gt_bbox, cfg: AegisConfig):
    center_logits = out['center_logits']
    B, _, H, W = center_logits.shape
    device = gt_bbox.device
    target_center = make_center_targets(gt_bbox, H, W, cfg.search_size, cfg)

    cx = gt_bbox[:, 0] + gt_bbox[:, 2] / 2
    cy = gt_bbox[:, 1] + gt_bbox[:, 3] / 2
    gx = cx / cfg.search_size * W
    gy = cy / cfg.search_size * H
    ix = torch.clamp(gx.long(), 0, W - 1)
    iy = torch.clamp(gy.long(), 0, H - 1)
    batch_idx = torch.arange(B, device=device)
    feature_stride = float(cfg.search_size) / float(W)

    # UETrack-like dense classification/task losses.
    l_center = focal_bce_logits(center_logits, target_center, alpha=0.25, gamma=2.0)
    l_obj = focal_bce_logits(out['objectness_logits'], target_center, alpha=0.35, gamma=2.0)

    logits_flat = out['response_logits'][:, 0].flatten(1)
    gt_index = iy * W + ix
    l_task = F.cross_entropy(logits_flat, gt_index)

    # Center-size head: dx/dy + log(w/h). This is the key fix for tiny UAVs.
    pred_offset = out['offset'][batch_idx, :, iy, ix]
    target_offset = torch.stack([gx - (ix.float() + 0.5), gy - (iy.float() + 0.5)], dim=-1).clamp(-0.5, 0.5)
    l_offset = F.smooth_l1_loss(pred_offset, target_offset)

    pred_log_wh = out['log_size'][batch_idx, :, iy, ix]
    target_log_wh = torch.log((gt_bbox[:, 2:4] / feature_stride).clamp_min(0.05))
    l_logwh = F.smooth_l1_loss(pred_log_wh, target_log_wh)

    pred_cx = (ix.float() + 0.5 + pred_offset[:, 0]) * feature_stride
    pred_cy = (iy.float() + 0.5 + pred_offset[:, 1]) * feature_stride
    pred_w = torch.exp(pred_log_wh[:, 0]).clamp(0.05, cfg.search_size) * feature_stride
    pred_h = torch.exp(pred_log_wh[:, 1]).clamp(0.05, cfg.search_size) * feature_stride
    pred_box = torch.stack([pred_cx - pred_w / 2, pred_cy - pred_h / 2, pred_w, pred_h], dim=-1)

    l_l1 = F.smooth_l1_loss(pred_box, gt_bbox)
    giou_vec = giou_loss_xyxy(xywh_to_xyxy_t(pred_box), xywh_to_xyxy_t(gt_bbox))
    l_giou = giou_vec.mean()

    with torch.no_grad():
        iou = box_iou_xyxy(xywh_to_xyxy_t(pred_box), xywh_to_xyxy_t(gt_bbox)).clamp(0, 1)
        quality_target = torch.zeros_like(target_center)
        quality_target[batch_idx, 0, iy, ix] = iou
    l_quality = focal_bce_logits(out['quality_logits'], quality_target, alpha=0.5, gamma=2.0)

    l_corr = torch.zeros((), device=device)
    if 'corr' in out and out['corr'] is not None:
        l_corr = F.binary_cross_entropy(out['corr'].clamp(1e-5, 1 - 1e-5), target_center.clamp(0, 1))

    # The bbox branch is trained directly, but not allowed to overwhelm response learning.
    total = (
        l_task
        + l_center
        + 0.25 * l_obj
        + 0.25 * l_quality
        + 0.25 * l_corr
        + 2.0 * l_giou
        + 1.0 * l_l1
        + 1.0 * l_offset
        + 1.0 * l_logwh
    )
    with torch.no_grad():
        wr = (pred_w / gt_bbox[:, 2].clamp_min(1e-6)).mean()
        hr = (pred_h / gt_bbox[:, 3].clamp_min(1e-6)).mean()
    return {
        'loss': total,
        'task': l_task,
        'center': l_center,
        'corr': l_corr,
        'giou': l_giou,
        'l1': l_l1,
        'offset': l_offset,
        'logwh': l_logwh,
        'obj': l_obj,
        'quality': l_quality,
        'box': 2.0 * l_giou + l_l1 + l_offset + l_logwh,
        'wr': wr.detach(),
        'hr': hr.detach(),
        'pred_w_mean': pred_w.mean().detach(),
        'pred_h_mean': pred_h.mean().detach(),
        'gt_w_mean': gt_bbox[:, 2].mean().detach(),
        'gt_h_mean': gt_bbox[:, 3].mean().detach(),
    }


def ranking_margin_loss(pos_scores, neg_scores, margin=0.2):
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return torch.zeros((), device=pos_scores.device if pos_scores.numel() else neg_scores.device)
    return torch.relu(margin - pos_scores[:, None] + neg_scores[None, :]).mean()
