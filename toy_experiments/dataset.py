import math
import torch


def eight_gaussian_centers(device, scale=4.0):
    return torch.tensor([
        [1, 0],
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [0, 1],
        [-1 / math.sqrt(2), 1 / math.sqrt(2)],
        [-1, 0],
        [-1 / math.sqrt(2), -1 / math.sqrt(2)],
        [0, -1],
        [1 / math.sqrt(2), -1 / math.sqrt(2)],
    ], device=device).float() * scale


def T_8gaussians(z, scale=4.0, std=0.25):
    angles = torch.atan2(z[:, 1], z[:, 0])
    sector = (((angles + math.pi) / (2 * math.pi)) * 8).long() % 8

    centers = eight_gaussian_centers(z.device, scale=scale)

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


def T_circles(z, noise_std=0.06):
    angle = 2 * math.pi * torch.sigmoid(z[:, 0])
    radius = torch.where(z[:, 1] > 0, 4.0, 2.0)
    circle = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
    radial_noise = noise_std * z[:, 1:2] * torch.stack(
        [torch.cos(angle), torch.sin(angle)],
        dim=1,
    )
    return circle + radial_noise


def sample_circles(batch_size, device="cuda", noise_std=0.06):
    angle = 2 * math.pi * torch.rand(batch_size, device=device)
    radius = torch.where(
        torch.rand(batch_size, device=device) < 0.5,
        torch.full((batch_size,), 2.0, device=device),
        torch.full((batch_size,), 4.0, device=device),
    )
    circle = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
    radial = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    tangent = torch.stack([-torch.sin(angle), torch.cos(angle)], dim=1)
    noise = noise_std * torch.randn(batch_size, 1, device=device) * radial
    noise = noise + noise_std * torch.randn(batch_size, 1, device=device) * tangent
    return circle + noise


def T_pinwheel(z, num_classes=5, radial_std=0.3, tangential_std=0.1, rate=0.25, scale=2.0):
    labels = (((torch.atan2(z[:, 1], z[:, 0]) + math.pi) / (2 * math.pi)) * num_classes).long()
    labels = labels % num_classes
    features = torch.stack(
        [
            1.0 + radial_std * z[:, 0],
            tangential_std * z[:, 1],
        ],
        dim=1,
    )
    angles = 2 * math.pi * labels.float() / num_classes + rate * torch.exp(features[:, 0])
    sin_angle = torch.sin(angles)
    cos_angle = torch.cos(angles)
    x = scale * (features[:, 0] * cos_angle - features[:, 1] * sin_angle)
    y = scale * (features[:, 0] * sin_angle + features[:, 1] * cos_angle)
    return torch.stack([x, y], dim=1)


def sample_pinwheel(
    batch_size,
    device="cuda",
    num_classes=5,
    radial_std=0.3,
    tangential_std=0.1,
    rate=0.25,
    scale=2.0,
):
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    features = torch.randn(batch_size, 2, device=device)
    features = features * torch.tensor([radial_std, tangential_std], device=device)
    features = features + torch.tensor([1.0, 0.0], device=device)

    angles = 2 * math.pi * labels.float() / num_classes + rate * torch.exp(features[:, 0])
    sin_angle = torch.sin(angles)
    cos_angle = torch.cos(angles)
    x = scale * (features[:, 0] * cos_angle - features[:, 1] * sin_angle)
    y = scale * (features[:, 0] * sin_angle + features[:, 1] * cos_angle)
    return torch.stack([x, y], dim=1)


def sample_data(batch_size, dataset="8gaussians", device="cuda"):
    if dataset == "8gaussians":
        centers = eight_gaussian_centers(device)
        mode = torch.randint(0, len(centers), (batch_size,), device=device)
        return centers[mode] + 0.25 * torch.randn(batch_size, 2, device=device)
    if dataset == "circles":
        return sample_circles(batch_size, device=device)
    if dataset == "pinwheel":
        return sample_pinwheel(batch_size, device=device)

    z = torch.randn(batch_size, 2, device=device)
    if dataset == "swissroll":
        return T_swissroll(z)
    if dataset == "twomoons":
        return T_twomoons(z)

    raise ValueError(dataset)


def sample_pair(batch_size, dataset="8gaussians", device="cuda", coupling="independent"):
    x0 = torch.randn(batch_size, 2, device=device)

    if coupling == "independent":
        x1 = sample_data(batch_size, dataset=dataset, device=device)
    elif coupling == "deterministic":
        if dataset == "8gaussians":
            x1 = T_8gaussians(x0)
        elif dataset == "swissroll":
            x1 = T_swissroll(x0)
        elif dataset == "twomoons":
            x1 = T_twomoons(x0)
        elif dataset == "circles":
            x1 = T_circles(x0)
        elif dataset == "pinwheel":
            x1 = T_pinwheel(x0)
        else:
            raise ValueError(dataset)
    else:
        raise ValueError(f"Unknown coupling: {coupling}")

    return x0, x1
