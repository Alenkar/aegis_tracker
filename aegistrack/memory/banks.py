from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from ..config import AegisConfig


class PrototypeBank:
    def __init__(self, max_size: int, ema: float = 0.95, device: str = 'cpu'):
        self.max_size = max_size
        self.ema = ema
        self.device = device
        self.items: List[torch.Tensor] = []
        self.meta: List[Dict] = []

    def clear(self):
        self.items.clear(); self.meta.clear()

    def write(self, emb: torch.Tensor, meta: Optional[Dict] = None, ema_update: bool = False):
        e = F.normalize(emb.detach().flatten().to(self.device), dim=0)
        if ema_update and self.items:
            self.items[-1] = F.normalize(self.ema * self.items[-1] + (1 - self.ema) * e, dim=0)
            if meta: self.meta[-1].update(meta)
            return
        self.items.append(e); self.meta.append(meta or {})
        while len(self.items) > self.max_size:
            self.items.pop(0); self.meta.pop(0)

    def max_similarity(self, emb: Optional[torch.Tensor]) -> float:
        if emb is None or not self.items:
            return 0.0
        e = F.normalize(emb.detach().flatten().to(self.device), dim=0)
        return max(float(torch.dot(e, m).clamp(-1, 1).item()) for m in self.items)

    def mean_topk_similarity(self, emb: Optional[torch.Tensor], k: int = 3) -> float:
        if emb is None or not self.items:
            return 0.0
        e = F.normalize(emb.detach().flatten().to(self.device), dim=0)
        sims = sorted([float(torch.dot(e, m).clamp(-1, 1).item()) for m in self.items], reverse=True)
        return float(sum(sims[:min(k, len(sims))]) / max(1, min(k, len(sims))))


class AegisMemory:
    def __init__(self, cfg: AegisConfig):
        self.cfg = cfg
        self.initial = PrototypeBank(1, cfg.memory_ema, cfg.device)
        self.stable = PrototypeBank(cfg.stable_memory_size, cfg.memory_ema, cfg.device)
        self.recent = PrototypeBank(cfg.recent_memory_size, cfg.memory_ema, cfg.device)
        self.negative = PrototypeBank(cfg.negative_memory_size, cfg.memory_ema, cfg.device)
        self.distractor = PrototypeBank(cfg.distractor_memory_size, cfg.memory_ema, cfg.device)
        self.quarantine = PrototypeBank(cfg.quarantine_memory_size, cfg.memory_ema, cfg.device)

    def clear(self):
        for b in (self.initial, self.stable, self.recent, self.negative, self.distractor, self.quarantine):
            b.clear()

    def identity_parts(self, emb: Optional[torch.Tensor]):
        return {
            'init': self.initial.max_similarity(emb),
            'stable': self.stable.mean_topk_similarity(emb, 3),
            'recent': self.recent.mean_topk_similarity(emb, 3),
            'negative': self.negative.max_similarity(emb),
            'distractor': self.distractor.max_similarity(emb),
        }

    def identity_score(self, emb: Optional[torch.Tensor]):
        p = self.identity_parts(emb)
        raw = 0.30 * p['init'] + 0.25 * p['stable'] + 0.10 * p['recent'] - 0.20 * p['negative'] - 0.15 * p['distractor']
        return float(max(0.0, min(1.0, 0.5 + raw))), p
