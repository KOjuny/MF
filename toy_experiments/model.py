import torch
import torch.nn as nn


class FlowMatchingMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=128, out_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]

        h = torch.cat([x, t], dim=1)
        return self.net(h)


class MeanFlowMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=128, out_dim=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, r, t):
        if r.ndim == 1:
            r = r[:, None]
        if t.ndim == 1:
            t = t[:, None]

        h = torch.cat([x, r, t], dim=1)
        return self.net(h)