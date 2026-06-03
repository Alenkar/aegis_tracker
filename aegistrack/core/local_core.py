from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AegisConfig
from ..utils.image import crop_with_context, crop_to_tensor, crop_point_to_frame
from ..utils.box_ops import make_bbox_from_center, bbox_area
from ..candidates.types import Candidate, CandidateSource, LocalOutput, TrackState, BBox


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.a = ConvBNAct(c, c, 3, 1)
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.b(self.a(x)))


class HighResTinyBackbone(nn.Module):
    def __init__(self, in_chans: int = 3, dim: int = 128, stride: int = 4):
        super().__init__()
        if stride not in (4, 8):
            raise ValueError("local_feature_stride must be 4 or 8")
        c1 = max(32, dim // 2)
        self.stem = nn.Sequential(
            ConvBNAct(in_chans, c1, 3, 2),
            ConvBNAct(c1, dim, 3, 2),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.down8 = nn.Sequential(ConvBNAct(dim, dim, 3, 2), ResBlock(dim)) if stride == 8 else nn.Identity()
        self.out_norm = nn.GroupNorm(8, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.down8(self.stem(x)))


class TokenInteraction(nn.Module):
    """Lightweight UETrack-like target-conditioned interaction.

    This is not a copy of UETrack code. It implements the required principle inside
    this project: template/search tokens interact before the prediction head, and
    bbox is predicted from the same target-aware feature map as the response.
    """
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        heads = max(1, min(heads, dim // 32))
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.GroupNorm(8, dim)
        self.ffn = nn.Sequential(ConvBNAct(dim, dim * 2, 1, 1), nn.Conv2d(dim * 2, dim, 1))
        self.heads = heads
        self.dim = dim

    def forward(self, zf: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        B, C, H, W = xf.shape
        z = F.adaptive_avg_pool2d(zf, (8, 8))
        q = self.q(xf).flatten(2).transpose(1, 2)       # B,N,C
        k = self.k(z).flatten(2).transpose(1, 2)        # B,M,C
        v = self.v(z).flatten(2).transpose(1, 2)        # B,M,C
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(float(C)), dim=-1)
        ctx = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        y = self.norm(xf + self.proj(ctx))
        return self.norm(y + self.ffn(y))


class LocalCore(nn.Module):
    """UETrack-like short-term core for Aegis.

    Main change versus previous Aegis: bbox is a first-class prediction-head output.
    It is center-size regression (offset + log w/h), not FCOS ltrb and not stable-size.
    This avoids stride-imposed lower bounds and lets boxes shrink/enlarge for tiny UAVs.
    """

    def __init__(self, cfg: AegisConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_stride = int(getattr(cfg, 'local_feature_stride', 4))
        d = int(getattr(cfg, 'local_feature_dim', cfg.embed_dim))
        self.dim = d
        self.backbone = HighResTinyBackbone(cfg.in_chans, d, self.feature_stride)

        hidden = max(32, d // 2)
        self.target_gate = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, d), nn.Sigmoid(),
        )
        self.token_interaction = TokenInteraction(d)
        self.fuse = nn.Sequential(ConvBNAct(2 * d + 1, d, 3, 1), ResBlock(d), ResBlock(d))
        self.pred = nn.Sequential(ConvBNAct(d, d, 3, 1), ResBlock(d))
        self.center_head = nn.Conv2d(d, 1, 1)
        self.obj_head = nn.Conv2d(d, 1, 1)
        self.quality_head = nn.Conv2d(d, 1, 1)
        self.offset_head = nn.Conv2d(d, 2, 1)
        self.log_size_head = nn.Conv2d(d, 2, 1)
        # Start around 2 feature cells, but exp(log_size) can go below one cell.
        nn.init.constant_(self.log_size_head.bias, math.log(2.0))
        nn.init.zeros_(self.offset_head.bias)
        self.token_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.template_tensor: Optional[torch.Tensor] = None
        self.initial_template_tensor: Optional[torch.Tensor] = None
        self.to(cfg.device)

    def initialize(self, frame, init_bbox: BBox):
        side = max(init_bbox[2], init_bbox[3]) * 4.0 + 16.0
        crop, _ = crop_with_context(frame, init_bbox, side)
        tensor = crop_to_tensor(crop, self.cfg.template_size, self.cfg.device)
        self.initial_template_tensor = tensor.detach().clone()
        self.template_tensor = tensor.detach().clone()

    def runtime_template(self) -> torch.Tensor:
        if self.initial_template_tensor is None:
            raise RuntimeError('LocalCore is not initialized. Call initialize(frame, bbox).')
        return self.initial_template_tensor

    def _template_region(self, zf: torch.Tensor) -> torch.Tensor:
        B, C, H, W = zf.shape
        k = int(getattr(self.cfg, 'template_kernel_size', 5))
        k = max(3, k | 1)
        k = min(k, H if H % 2 == 1 else H - 1, W if W % 2 == 1 else W - 1)
        cy, cx = H // 2, W // 2
        r = k // 2
        return zf[:, :, cy - r:cy + r + 1, cx - r:cx + r + 1]

    def _template_proto(self, zf: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._template_region(zf).mean(dim=(2, 3)), dim=1)

    def _multi_token_corr(self, zf: torch.Tensor, xf: torch.Tensor):
        B, C, H, W = xf.shape
        patch = self._template_region(zf)
        proto = F.normalize(patch.mean(dim=(2, 3)), dim=1)
        search = F.normalize(xf, dim=1)
        kernels = F.normalize(patch.flatten(2), dim=2).view_as(patch)
        pad = patch.shape[-1] // 2
        outs = []
        for b in range(B):
            outs.append(F.conv2d(search[b:b + 1], kernels[b:b + 1].transpose(0, 1), padding=pad, groups=C))
        corr_feat = torch.cat(outs, dim=0) / math.sqrt(float(patch.shape[-1] * patch.shape[-2]))
        corr_feat = torch.tanh(corr_feat)
        corr_prior = ((corr_feat.mean(dim=1, keepdim=True) + 1.0) * 0.5).clamp(1e-4, 1.0)
        return proto, corr_feat, corr_prior

    def _fuse_template_search(self, template: torch.Tensor, search: torch.Tensor):
        zf = self.backbone(template)
        xf = self.backbone(search)
        xf = self.token_interaction(zf, xf)
        proto, corr_feat, corr_prior = self._multi_token_corr(zf, xf)
        gate = self.target_gate(proto).view(proto.shape[0], proto.shape[1], 1, 1)
        gate_min = float(getattr(self.cfg, 'target_gate_min', 0.25))
        xf_gated = xf * (gate_min + (1.0 - gate_min) * gate)
        fmap = self.fuse(torch.cat([xf_gated, corr_feat, corr_prior], dim=1))
        return zf, xf, proto, corr_feat, corr_prior, fmap

    def forward_train(self, template: torch.Tensor, search: torch.Tensor) -> Dict[str, torch.Tensor]:
        zf, xf, proto, corr_feat, corr_prior, fmap = self._fuse_template_search(template, search)
        p = self.pred(fmap)
        center_logits = self.center_head(p)
        objectness_logits = self.obj_head(p)
        quality_logits = self.quality_head(p)
        offset = torch.tanh(self.offset_head(p)) * 0.5
        log_size = self.log_size_head(p).clamp(-4.0, 4.0)
        response_logits = center_logits + 0.5 * objectness_logits + 0.5 * quality_logits
        if corr_prior is not None and float(getattr(self.cfg, 'corr_response_weight', 0.0)) > 0:
            response_logits = response_logits + float(getattr(self.cfg, 'corr_response_weight', 0.35)) * torch.log(corr_prior.clamp_min(1e-4))
        target_token = F.normalize(self.token_proj(proto), dim=-1)
        token_map = F.normalize(self.token_proj(p.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), dim=1)
        return {
            'center_logits': center_logits,
            'objectness_logits': objectness_logits,
            'quality_logits': quality_logits,
            'center': torch.sigmoid(center_logits),
            'objectness': torch.sigmoid(objectness_logits),
            'quality': torch.sigmoid(quality_logits),
            'offset': offset,
            'log_size': log_size,
            'size': torch.exp(log_size),  # feature-cell units, compatibility only
            'response_logits': response_logits,
            'response': torch.sigmoid(response_logits),
            'corr': corr_prior,
            'corr_feat': corr_feat,
            'fmap': p,
            'target_token': target_token,
            'token_map': token_map,
            'template_features': zf,
            'search_features': xf,
        }

    @torch.no_grad()
    def encode_target(self, frame, bbox: BBox) -> torch.Tensor:
        side = max(bbox[2], bbox[3]) * 4.0 + 16.0
        crop, _ = crop_with_context(frame, bbox, side)
        x = crop_to_tensor(crop, self.cfg.template_size, self.cfg.device)
        f = self.backbone(x)
        proto = self._template_proto(f)
        return F.normalize(self.token_proj(proto)[0], dim=0)

    def fuse_response(self, out: Dict[str, torch.Tensor], state='TRACKING'):
        logits = out['response_logits']
        H, W = logits.shape[-2:]
        if state == 'TRACKING':
            w = float(getattr(self.cfg, 'window_weight_tracking', 0.10))
        else:
            w = float(getattr(self.cfg, 'window_weight_lost', 0.0))
        if w > 0:
            wy = torch.hann_window(H, device=logits.device)
            wx = torch.hann_window(W, device=logits.device)
            window = torch.outer(wy, wx).view(1, 1, H, W).clamp_min(1e-6)
            logits = logits + torch.log(((1.0 - w) + w * window).clamp_min(1e-6))
        return torch.sigmoid(logits)

    @torch.no_grad()
    def forward_local(self, frame, crop, crop_meta: Dict[str, float], stable_size: Tuple[float, float], state: TrackState) -> LocalOutput:
        template = self.runtime_template()
        search = crop_to_tensor(crop, self.cfg.search_size, self.cfg.device)
        out = self.forward_train(template, search)
        response = self.fuse_response(out, state.value)[0, 0]
        k = self.cfg.topk_for_state(state.value)
        cands = self.decode_topk(response, out, crop_meta, k)
        best = cands[0] if cands else Candidate((0, 0, stable_size[0], stable_size[1]), CandidateSource.LOCAL)
        raw_score = float(response.max().item())
        return LocalOutput(
            best_bbox=best.bbox,
            center_map=out['center'][0, 0],
            objectness_map=out['objectness'][0, 0],
            quality_map=out['quality'][0, 0],
            size_map=out['size'][0],
            feature_map=out['fmap'][0],
            response_map=response,
            topk_candidates=cands,
            target_token=out['target_token'][0],
            raw_score=raw_score,
            windowed_score=raw_score,
            center_good=raw_score > float(getattr(self.cfg, 'min_raw_score', 0.20)),
            size_bad=False,
            crop_meta=crop_meta,
        )

    def _response_metrics(self, response: torch.Tensor, iy: int, ix: int):
        H, W = response.shape
        peak = float(response[iy, ix].item())
        mask = torch.ones_like(response, dtype=torch.bool)
        r = 2
        mask[max(0, iy-r):min(H, iy+r+1), max(0, ix-r):min(W, ix+r+1)] = False
        bg = response[mask]
        second = float(bg.max().item()) if bg.numel() else 0.0
        margin = peak - second
        ratio = peak / max(second, 1e-6)
        psr = (peak - float(bg.mean().item())) / max(float(bg.std().item()), 1e-6) if bg.numel() > 4 else 0.0
        return second, margin, ratio, psr

    def _peak_window_weights(self, response: torch.Tensor, iy: int, ix: int, window: int):
        radius = max(0, window // 2)
        y1, y2 = max(0, iy - radius), min(response.shape[0], iy + radius + 1)
        x1, x2 = max(0, ix - radius), min(response.shape[1], ix + radius + 1)
        weights = response[y1:y2, x1:x2].detach().clamp_min(1e-6)
        weights = weights / weights.sum().clamp_min(1e-6)
        return y1, y2, x1, x2, weights

    def _decode_center_at_peak(
        self,
        response: torch.Tensor,
        offset: torch.Tensor,
        iy: int,
        ix: int,
        feature_stride: float,
    ) -> tuple[float, float]:
        window = int(getattr(self.cfg, 'bbox_decode_center_window', 1))
        if window <= 1:
            dx = float(offset[0, iy, ix].item())
            dy = float(offset[1, iy, ix].item())
            return (ix + 0.5 + dx) * feature_stride, (iy + 0.5 + dy) * feature_stride

        y1, y2, x1, x2, weights = self._peak_window_weights(response, iy, ix, window)
        yy, xx = torch.meshgrid(
            torch.arange(y1, y2, device=response.device, dtype=response.dtype),
            torch.arange(x1, x2, device=response.device, dtype=response.dtype),
            indexing='ij',
        )
        center_x = (xx + 0.5 + offset[0, y1:y2, x1:x2]) * feature_stride
        center_y = (yy + 0.5 + offset[1, y1:y2, x1:x2]) * feature_stride
        return float((center_x * weights).sum().item()), float((center_y * weights).sum().item())

    def _decode_log_size_at_peak(self, response: torch.Tensor, log_size: torch.Tensor, iy: int, ix: int) -> tuple[float, float]:
        window = int(getattr(self.cfg, 'bbox_decode_size_window', 1))
        if window <= 1:
            return float(log_size[0, iy, ix].item()), float(log_size[1, iy, ix].item())

        y1, y2, x1, x2, weights = self._peak_window_weights(response, iy, ix, window)
        size_patch = log_size[:, y1:y2, x1:x2]
        decoded = (size_patch * weights.unsqueeze(0)).sum(dim=(1, 2))
        return float(decoded[0].item()), float(decoded[1].item())

    def decode_topk(self, response: torch.Tensor, out: Dict[str, torch.Tensor], crop_meta, k: int) -> List[Candidate]:
        H, W = response.shape
        vals, idxs = torch.topk(response.flatten(), k=min(k, response.numel()))
        offset = out['offset'][0]
        log_size = out['log_size'][0]
        token_map = out['token_map'][0]
        obj = out['objectness'][0, 0]
        qual = out['quality'][0, 0]
        feature_stride = self.cfg.search_size / float(W)
        scale = crop_meta['side'] / float(self.cfg.search_size)
        out_cands: List[Candidate] = []
        for val, idx in zip(vals, idxs):
            iy = int(idx.item() // W)
            ix = int(idx.item() % W)
            # Decode center from the local response mass, not one raw top-k cell.
            px, py = self._decode_center_at_peak(response, offset, iy, ix, feature_stride)
            cx, cy = crop_point_to_frame(px, py, crop_meta, self.cfg.search_size)
            # log_size is in feature-cell units; average a small response-weighted
            # neighborhood to avoid bbox boundary jitter when adjacent peaks swap.
            log_w, log_h = self._decode_log_size_at_peak(response, log_size, iy, ix)
            pred_w_search = float(math.exp(log_w)) * feature_stride
            pred_h_search = float(math.exp(log_h)) * feature_stride
            pred_w = float(np.clip(pred_w_search * scale, 1.0, crop_meta['side'] * 0.8))
            pred_h = float(np.clip(pred_h_search * scale, 1.0, crop_meta['side'] * 0.8))
            bbox = make_bbox_from_center(cx, cy, pred_w, pred_h)
            emb = token_map[:, iy, ix]
            second, margin, ratio, psr = self._response_metrics(response, iy, ix)
            score = float(val.item())
            c = Candidate(
                bbox=bbox,
                source=CandidateSource.TOPK,
                local_score=score,
                center_score=float(response[iy, ix].item()),
                objectness=float(obj[iy, ix].item()),
                quality=float(qual[iy, ix].item()),
                final_score=score,
                visual_emb=emb,
            )
            c.reason += [f'second={second:.3f}', f'margin={margin:.3f}', f'ratio={ratio:.2f}', f'psr={psr:.2f}', 'bbox=predicted']
            # dynamic attributes consumed by runtime/logger.
            c.second_score = second
            c.peak_margin = margin
            c.peak_ratio = ratio
            c.psr = psr
            c.predicted_size = (pred_w, pred_h)
            c.bbox_predicted = bbox
            c.bbox_decode_center_window = int(getattr(self.cfg, 'bbox_decode_center_window', 1))
            c.bbox_decode_size_window = int(getattr(self.cfg, 'bbox_decode_size_window', 1))
            out_cands.append(c)
        return out_cands
