import math
import torch


def T_8gaussians(z, scale=4.0, std=0.25):
    angles = torch.atan2(z[:, 1], z[:, 0])
    sector = (((angles + math.pi) / (2 * math.pi)) * 8).long() % 8

    centers = torch.tensor([
        [1, 0],
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [0, 1],
        [-1 / math.sqrt(2), 1 / math.sqrt(2)],
        [-1, 0],
        [-1 / math.sqrt(2), -1 / math.sqrt(2)],
        [0, -1],
        [1 / math.sqrt(2), -1 / math.sqrt(2)],
    ], device=z.device).float() * scale

    return centers[sector] + std * z


def T_swissroll(z, noise_std=0.04):
    u = torch.sigmoid(z[:, 0])
    theta = 0.5 * math.pi + 3.5 * math.pi * u
    radius = 0.18 * theta

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    curve = torch.stack([x, y], dim=1)

    tangent = torch.stack([
        0.18 * torch.cos(theta) - radius * torch.sin(theta),
        0.18 * torch.sin(theta) + radius * torch.cos(theta),
    ], dim=1)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=1)
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-8)

    return curve + noise_std * z[:, 1:2] * normal


def T_twomoons(z):
    u = torch.sigmoid(z[:, 0])
    theta = math.pi * u

    upper = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    lower = torch.stack([1 - torch.cos(theta), -torch.sin(theta) - 0.5], dim=1)

    mask = (z[:, 1] > 0).float().unsqueeze(1)
    x = mask * upper + (1 - mask) * lower
    return x + 0.05 * z


def sample_pair(batch_size, dataset="8gaussians", device="cuda"):
    x0 = torch.randn(batch_size, 2, device=device)

    if dataset == "8gaussians":
        x1 = T_8gaussians(x0)
    elif dataset == "swissroll":
        x1 = T_swissroll(x0)
    elif dataset == "twomoons":
        x1 = T_twomoons(x0)
    else:
        raise ValueError(dataset)

    return x0, x1
