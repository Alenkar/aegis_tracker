from __future__ import annotations

from collections import deque
from typing import Deque

import torch
import torch.nn.functional as F


def _as_token(token: torch.Tensor) -> torch.Tensor:
    return F.normalize(token.detach().flatten().float().cpu(), dim=0)


class TargetTemporalMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stable: Deque[torch.Tensor] = deque(maxlen=int(getattr(cfg, 'memory_stable_size', 16)))
        self.recent: Deque[torch.Tensor] = deque(maxlen=int(getattr(cfg, 'memory_recent_size', 8)))
        self.distractors: Deque[torch.Tensor] = deque(maxlen=int(getattr(cfg, 'memory_distractor_size', 24)))
        self._confirm_count = 0

    def reset(self, init_token: torch.Tensor | None = None) -> None:
        self.stable.clear()
        self.recent.clear()
        self.distractors.clear()
        self._confirm_count = 0
        if init_token is not None:
            self.recent.append(_as_token(init_token))

    def __len__(self) -> int:
        return len(self.stable) + len(self.recent)

    def _tokens(self) -> list[torch.Tensor]:
        return list(self.stable) + list(self.recent)

    def match(self, token: torch.Tensor | None) -> dict[str, float]:
        if token is None or len(self) == 0:
            return {
                'memory_score': 0.0,
                'memory_q25': 1.0,
                'memory_q50': 1.0,
                'memory_q75': 1.0,
                'distractor_penalty': 0.0,
                'memory_count': float(len(self)),
                'stable_memory_count': float(len(self.stable)),
                'recent_memory_count': float(len(self.recent)),
                'distractor_count': float(len(self.distractors)),
            }

        q = _as_token(token)
        mem = torch.stack(self._tokens(), dim=0)
        distances = (1.0 - torch.mv(mem, q).clamp(-1.0, 1.0)).clamp(0.0, 2.0)
        qs = torch.quantile(distances, torch.tensor([0.25, 0.50, 0.75]))
        q25, q50, q75 = [float(v.item()) for v in qs]
        memory_score = max(0.0, min(1.0, 1.0 - q50))

        distractor_penalty = 0.0
        if self.distractors:
            dis = torch.stack(list(self.distractors), dim=0)
            distractor_penalty = float(torch.mv(dis, q).max().clamp(0.0, 1.0).item())

        return {
            'memory_score': memory_score,
            'memory_q25': q25,
            'memory_q50': q50,
            'memory_q75': q75,
            'distractor_penalty': distractor_penalty,
            'memory_count': float(len(self)),
            'stable_memory_count': float(len(self.stable)),
            'recent_memory_count': float(len(self.recent)),
            'distractor_count': float(len(self.distractors)),
        }

    def should_update(self, token: torch.Tensor | None, tracking_score: float, response_shape_score: float) -> tuple[bool, dict[str, float]]:
        stats = self.match(token)
        if token is None:
            return False, stats
        if tracking_score < float(getattr(self.cfg, 'memory_update_min_tracking_score', 0.55)):
            return False, stats
        if response_shape_score < float(getattr(self.cfg, 'memory_update_min_shape_score', 0.55)):
            return False, stats
        if stats['memory_q25'] > float(getattr(self.cfg, 'memory_update_max_q25', 0.35)):
            return False, stats
        if stats['memory_q50'] > float(getattr(self.cfg, 'memory_update_max_q50', 0.45)):
            return False, stats
        if stats['memory_q75'] > float(getattr(self.cfg, 'memory_update_max_q75', 0.60)):
            return False, stats
        return True, stats

    def add_confirmed(self, token: torch.Tensor) -> dict[str, float]:
        token = _as_token(token)
        self.recent.append(token)
        self._confirm_count += 1
        promote_every = int(getattr(self.cfg, 'memory_stable_promote_frames', 3))
        if self._confirm_count >= max(1, promote_every):
            self.stable.append(token)
            self._confirm_count = 0
        return self.match(token)

    def add_distractors(self, candidates, accepted=None) -> None:
        if not bool(getattr(self.cfg, 'use_temporal_memory', True)):
            return
        max_add = int(getattr(self.cfg, 'memory_distractor_add_topk', 3))
        min_score = float(getattr(self.cfg, 'memory_distractor_min_local_score', 0.35))
        added = 0
        for cand in candidates:
            if cand is accepted:
                continue
            if added >= max_add:
                break
            if getattr(cand, 'visual_emb', None) is None:
                continue
            if float(getattr(cand, 'local_score', 0.0)) < min_score:
                continue
            self.distractors.append(_as_token(cand.visual_emb))
            added += 1
