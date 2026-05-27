import argparse
import math
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from dataset import sample_pair
from model import FlowMatchingMLP, MeanFlowMLP


@torch.no_grad()
def sample_oracle_transport(z, x_target, steps=100):
    times = torch.linspace(0.0, 1.0, steps + 1, device=z.device)
    trajectory = []

    for time in times:
        t = time.view(1, 1)
        x_t = (1.0 - t) * z + t * x_target
        trajectory.append(x_t)

    return x_target, torch.stack(trajectory)


@torch.no_grad()
def optimal_coupling(source, target):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "Optimal coupling requires scipy. Install scipy or use "
            "--gt-coupling dataset."
        ) from exc

    cost = torch.cdist(source, target).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    if not (row_ind == torch.arange(len(source)).numpy()).all():
        order = torch.as_tensor(row_ind).argsort()
        col_ind = col_ind[order.numpy()]

    return target[torch.as_tensor(col_ind, device=target.device)]


@torch.no_grad()
def sample_flow_matching(model, z, steps=100, return_trajectory=False):
    x = z.clone()
    dt = 1.0 / steps
    trajectory = [x.clone()] if return_trajectory else None

    for i in range(steps):
        t = torch.full((z.shape[0], 1), i / steps, device=z.device)
        v = model(x, t)
        x = x + dt * v
        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory)
    return x


@torch.no_grad()
def sample_meanflow(model, z, return_trajectory=False):
    r = torch.zeros(z.shape[0], 1, device=z.device)
    t = torch.ones(z.shape[0], 1, device=z.device)

    avg_v = model(z, r, t)
    x = z - avg_v
    if return_trajectory:
        return x, torch.stack([z.clone(), x.clone()])
    return x

@torch.no_grad()
def sample_meanflow_multistep(model, z, steps=10, return_trajectory=False):
    x = z.clone()
    trajectory = [x.clone()] if return_trajectory else None

    for i in range(steps):
        r = torch.full((z.shape[0], 1), (steps - i - 1) / steps, device=z.device)
        t = torch.full((z.shape[0], 1), (steps - i) / steps, device=z.device)

        avg_v = model(x, r, t)
        x = x - (t - r) * avg_v
        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory)
    return x


def filename_from_title(title):
    filename = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    return f"{filename}.png"


def save_or_show(title, save_dir=None, show=True):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename_from_title(title)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    if show:
        plt.show()
    else:
        plt.close()


def point_colors(n, total=None):
    total = n if total is None else total
    if total <= 1:
        values = [0.0]
    else:
        values = (torch.arange(n) / (total - 1)).tolist()
    return plt.get_cmap("turbo")(values)


def normalize(values):
    values = values.detach().cpu()
    low = values.min()
    high = values.max()
    span = (high - low).clamp_min(1e-8)
    return (values - low) / span


def colors_from_hsv(hue, saturation=0.78, value=0.9):
    hue = hue.detach().cpu() % 1.0
    hsv = torch.stack(
        [
            hue,
            torch.full_like(hue, saturation),
            torch.full_like(hue, value),
        ],
        dim=1,
    ).numpy()
    rgb = torch.as_tensor(hsv_to_rgb(hsv), dtype=torch.float32)
    alpha = torch.ones((len(hue), 1), dtype=rgb.dtype)
    return torch.cat([rgb, alpha], dim=1).numpy()


def position_colors(points, dataset):
    points = points.detach().cpu()
    x = points[:, 0]
    y = points[:, 1]

    if dataset == "8gaussians":
        angles = torch.atan2(y, x)
        sectors = (((angles + math.pi) / (2 * math.pi)) * 8).long() % 8
        radius_shade = 0.04 * (normalize(torch.linalg.norm(points, dim=1)) - 0.5)
        hue = sectors.float() / 8.0 + radius_shade
        return colors_from_hsv(hue, saturation=0.8, value=0.9)

    if dataset == "twomoons":
        upper = y > -0.25
        along_moon = normalize(x)
        hue = torch.where(
            upper,
            0.56 + 0.08 * along_moon,  # blue/cyan family
            0.04 + 0.08 * along_moon,  # red/orange family
        )
        return colors_from_hsv(hue, saturation=0.78, value=0.9)

    if dataset == "circles":
        radius = torch.linalg.norm(points, dim=1)
        inner = radius < radius.median()
        angle_color = normalize(torch.atan2(y, x))
        hue = torch.where(inner, 0.56 + 0.08 * angle_color, 0.03 + 0.08 * angle_color)
        return colors_from_hsv(hue, saturation=0.78, value=0.9)

    if dataset == "pinwheel":
        hue = normalize(torch.atan2(y, x))
        return colors_from_hsv(hue, saturation=0.82, value=0.9)

    if dataset == "swissroll":
        hue = normalize(torch.atan2(y, x)) * 0.75
        return colors_from_hsv(hue, saturation=0.75, value=0.9)

    return point_colors(len(points))


def point_indices(total, max_points):
    n = min(total, max_points)
    if n == total:
        return torch.arange(total)
    return torch.linspace(0, total - 1, n).long()


def compute_axis_limits(*tensors, pad_ratio=0.05):
    points = []
    for tensor in tensors:
        tensor = tensor.detach().cpu()
        points.append(tensor.reshape(-1, tensor.shape[-1])[:, :2])

    points = torch.cat(points, dim=0)
    xy_min = points.min(dim=0).values
    xy_max = points.max(dim=0).values
    center = (xy_min + xy_max) / 2
    span = (xy_max - xy_min).max()
    half = span * (0.5 + pad_ratio)
    if half == 0:
        half = torch.tensor(1.0)

    return (
        (center[0].item() - half.item(), center[0].item() + half.item()),
        (center[1].item() - half.item(), center[1].item() + half.item()),
    )


def set_axis_limits(axis_limits, ax=None):
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_aspect("equal")


def plot_samples(x, title, axis_limits=None, save_dir=None, show=True, colors=None):
    x = x.detach().cpu()
    colors = point_colors(len(x)) if colors is None else colors

    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=6, c=colors)
    if axis_limits is None:
        plt.axis("equal")
    else:
        set_axis_limits(axis_limits)
    plt.title(title)
    save_or_show(title, save_dir=save_dir, show=show)


def plot_coupling(
    z,
    x,
    title,
    max_lines=300,
    axis_limits=None,
    save_dir=None,
    show=True,
    colors=None,
):
    z = z.detach().cpu()
    x = x.detach().cpu()

    indices = point_indices(len(z), max_lines)
    colors = point_colors(len(z)) if colors is None else colors
    colors = colors[indices.numpy()]

    plt.figure(figsize=(5, 5))

    for color, idx in zip(colors, indices):
        plt.plot(
            [z[idx, 0], x[idx, 0]],
            [z[idx, 1], x[idx, 1]],
            alpha=0.15,
            linewidth=0.7,
            color=color,
        )

    plt.scatter(
        z[indices, 0],
        z[indices, 1],
        s=18,
        c=colors,
        marker="o",
        edgecolors="black",
        linewidths=0.25,
        label="source / noise",
    )
    plt.scatter(
        x[indices, 0],
        x[indices, 1],
        s=24,
        c=colors,
        marker="X",
        edgecolors="black",
        linewidths=0.25,
        label="target / generated",
    )

    if axis_limits is None:
        plt.axis("equal")
    else:
        set_axis_limits(axis_limits)
    plt.title(title)
    plt.legend()
    save_or_show(title, save_dir=save_dir, show=show)


def plot_trajectory(
    trajectory,
    title,
    max_lines=300,
    axis_limits=None,
    save_dir=None,
    show=True,
    colors=None,
):
    trajectory = trajectory.detach().cpu()
    indices = point_indices(trajectory.shape[1], max_lines)
    colors = point_colors(trajectory.shape[1]) if colors is None else colors
    colors = colors[indices.numpy()]

    plt.figure(figsize=(5, 5))

    for color, idx in zip(colors, indices):
        path = trajectory[:, idx]
        plt.plot(
            path[:, 0],
            path[:, 1],
            alpha=0.22,
            linewidth=0.8,
            color=color,
        )

    start = trajectory[0, indices]
    end = trajectory[-1, indices]
    plt.scatter(
        start[:, 0],
        start[:, 1],
        s=18,
        c=colors,
        marker="o",
        edgecolors="black",
        linewidths=0.25,
        label="source / noise",
    )
    plt.scatter(
        end[:, 0],
        end[:, 1],
        s=24,
        c=colors,
        marker="X",
        edgecolors="black",
        linewidths=0.25,
        label="target / generated",
    )

    if axis_limits is None:
        plt.axis("equal")
    else:
        set_axis_limits(axis_limits)
    plt.title(title)
    plt.legend()
    save_or_show(title, save_dir=save_dir, show=show)


def plot_trajectory_snapshots(
    trajectory,
    title,
    num_snapshots=6,
    max_points=2000,
    axis_limits=None,
    save_dir=None,
    show=True,
    colors=None,
):
    trajectory = trajectory.detach().cpu()
    indices = torch.linspace(0, trajectory.shape[0] - 1, num_snapshots).long()
    point_idx = point_indices(trajectory.shape[1], max_points)
    colors = point_colors(trajectory.shape[1]) if colors is None else colors
    colors = colors[point_idx.numpy()]

    fig, axes = plt.subplots(1, num_snapshots, figsize=(3 * num_snapshots, 3))
    if num_snapshots == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        x_t = trajectory[idx, point_idx]
        t = idx.item() / (trajectory.shape[0] - 1)
        marker = "o" if idx == indices[0] else "X" if idx == indices[-1] else "."
        size = 16 if marker == "o" else 22 if marker == "X" else 8
        ax.scatter(
            x_t[:, 0],
            x_t[:, 1],
            s=size,
            c=colors,
            marker=marker,
            edgecolors="black" if marker != "." else "none",
            linewidths=0.2 if marker != "." else 0.0,
        )
        if axis_limits is None:
            ax.set_aspect("equal")
        else:
            set_axis_limits(axis_limits, ax=ax)
        ax.set_title(f"t={t:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    save_or_show(title, save_dir=save_dir, show=show)


def coupling_metrics(z, x_gt, x_model):
    mse = ((x_model - x_gt) ** 2).sum(dim=1).mean()

    gt_disp = x_gt - z
    model_disp = x_model - z

    cos = torch.nn.functional.cosine_similarity(
        gt_disp, model_disp, dim=1
    ).mean()

    return {
        "mse": mse.item(),
        "cosine": cos.item(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory where visualization images will be saved.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./results/models"),
        help="Directory where trained model checkpoints are stored.",
    )
    parser.add_argument(
        "--dataset",
        default="8gaussians",
        choices=["8gaussians", "swissroll", "twomoons", "circles", "pinwheel"],
        help="Dataset to visualize.",
    )
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling the shared source noise.",
    )
    parser.add_argument(
        "--gt-coupling",
        choices=["optimal", "dataset"],
        default="optimal",
        help=(
            "Coupling used for ground-truth/oracle visualization and metrics. "
            "'optimal' solves a minimum-distance assignment; 'dataset' uses "
            "the dataset transform pairing z -> T(z)."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening interactive plot windows.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = args.dataset
    show = not args.no_show
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    n = args.n
    z, x_gt = sample_pair(n, dataset, device=device, coupling="deterministic")
    if args.gt_coupling == "optimal":
        x_gt_coupled = optimal_coupling(z, x_gt)
    else:
        x_gt_coupled = x_gt
    x_oracle, traj_oracle = sample_oracle_transport(z, x_gt_coupled, steps=100)
    sample_colors = position_colors(x_gt, dataset)
    coupling_colors = position_colors(x_gt_coupled, dataset)

    fm = FlowMatchingMLP().to(device)
    fm.load_state_dict(torch.load(args.model_dir / f"flow_matching_{dataset}.pt", map_location=device))
    fm.eval()

    rf = FlowMatchingMLP().to(device)
    rf.load_state_dict(torch.load(args.model_dir / f"rectified_flow_{dataset}.pt", map_location=device))
    rf.eval()

    mf = MeanFlowMLP().to(device)
    mf.load_state_dict(torch.load(args.model_dir / f"meanflow_{dataset}.pt", map_location=device))
    mf.eval()


    x_fm, traj_fm = sample_flow_matching(fm, z, steps=100, return_trajectory=True)
    x_rf, traj_rf = sample_flow_matching(rf, z, steps=100, return_trajectory=True)
    x_mf, traj_mf = sample_meanflow(mf, z, return_trajectory=True)
    x_mf_10, traj_mf_10 = sample_meanflow_multistep(mf, z, steps=10, return_trajectory=True)
    x_mf_50, traj_mf_50 = sample_meanflow_multistep(mf, z, steps=50, return_trajectory=True)

    axis_limits = compute_axis_limits(
        z,
        x_gt,
        x_gt_coupled,
        x_fm,
        x_rf,
        x_mf,
        x_mf_10,
        x_mf_50,
        traj_oracle,
        traj_fm,
        traj_rf,
        traj_mf,
        traj_mf_10,
        traj_mf_50,
    )

    plot_samples(x_gt, "Ground Truth Samples", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=sample_colors)
    plot_samples(x_oracle, "Oracle Transport Samples", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_samples(x_fm, "Flow Matching Samples", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_samples(x_rf, "Rectified Flow Samples", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_samples(x_mf, "MeanFlow Samples", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_samples(x_mf_10, "MeanFlow 10-step Samples", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_samples(x_mf_50, "MeanFlow 50-step Samples", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)

    plot_coupling(z, x_gt_coupled, "Ground Truth Coupling", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_trajectory(traj_oracle, "Oracle Transport Trajectory", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_trajectory_snapshots(
        traj_oracle,
        "Oracle Transport Snapshots",
        axis_limits=axis_limits,
        save_dir=args.save_dir,
        show=show,
        colors=coupling_colors,
    )
    plot_trajectory(traj_fm, "Flow Matching Trajectory", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_trajectory(traj_rf, "Rectified Flow Trajectory", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_trajectory(traj_mf, "MeanFlow Trajectory", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_trajectory(traj_mf_10, "MeanFlow 10-step Trajectory", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)
    plot_trajectory(traj_mf_50, "MeanFlow 50-step Trajectory", axis_limits=axis_limits, save_dir=args.save_dir, show=show, colors=coupling_colors)

    print("FM:", coupling_metrics(z, x_gt_coupled, x_fm))
    print("RF:", coupling_metrics(z, x_gt_coupled, x_rf))
    print("MeanFlow 1-step:", coupling_metrics(z, x_gt_coupled, x_mf))
    print("MeanFlow 10-step:", coupling_metrics(z, x_gt_coupled, x_mf_10))
    print("MeanFlow 50-step:", coupling_metrics(z, x_gt_coupled, x_mf_50))
