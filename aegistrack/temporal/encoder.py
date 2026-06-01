from __future__ import annotations
import torch.nn as nn


class TemporalRiskEncoder(nn.Module):
    """Trainable compact temporal identity/risk encoder."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 96):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.risk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 5), nn.Sigmoid(),
        )
        self.prior = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        y, h = self.gru(x)
        h = self.norm(h[-1])
        risks = self.risk(h)
        prior = self.prior(h)
        return {
            'hidden': h,
            'drift_risk': risks[:, 0],
            'switch_risk': risks[:, 1],
            'lost_risk': risks[:, 2],
            'update_risk': risks[:, 3],
            'recovery_risk': risks[:, 4],
            'bbox_prior_delta': prior,
        }
