import torch
import torch.nn as nn


def make_relu_mlp(in_dim, out_dim, hidden=256, depth=7):
    layers = []
    for i in range(depth):
        layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class FlowMatchingMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=256, out_dim=2, depth=7):
        super().__init__()

        self.net = make_relu_mlp(in_dim + 1, out_dim, hidden=hidden, depth=depth)

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]

        h = torch.cat([x, t], dim=1)
        return self.net(h)


class MeanFlowMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=256, out_dim=2, depth=7):
        super().__init__()

        self.net = make_relu_mlp(in_dim + 2, out_dim, hidden=hidden, depth=depth)

    def forward(self, x, r, t):
        if r.ndim == 1:
            r = r[:, None]
        if t.ndim == 1:
            t = t[:, None]

        h = torch.cat([x, r, t], dim=1)
        return self.net(h)
