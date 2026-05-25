import argparse
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt

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
    x = z + avg_v
    if return_trajectory:
        return x, torch.stack([z.clone(), x.clone()])
    return x

@torch.no_grad()
def sample_meanflow_multistep(model, z, steps=10, return_trajectory=False):
    x = z.clone()
    trajectory = [x.clone()] if return_trajectory else None

    for i in range(steps):
        r = torch.full((z.shape[0], 1), i / steps, device=z.device)
        t = torch.full((z.shape[0], 1), (i + 1) / steps, device=z.device)

        avg_v = model(x, r, t)
        x = x + (t - r) * avg_v
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


def plot_samples(x, title, save_dir=None, show=True):
    x = x.detach().cpu()

    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=4)
    plt.axis("equal")
    plt.title(title)
    save_or_show(title, save_dir=save_dir, show=show)


def plot_coupling(z, x, title, max_lines=300, save_dir=None, show=True):
    z = z.detach().cpu()
    x = x.detach().cpu()

    n = min(len(z), max_lines)

    plt.figure(figsize=(5, 5))

    for i in range(n):
        plt.plot(
            [z[i, 0], x[i, 0]],
            [z[i, 1], x[i, 1]],
            alpha=0.15,
            linewidth=0.7,
        )

    plt.scatter(z[:n, 0], z[:n, 1], s=5, label="noise")
    plt.scatter(x[:n, 0], x[:n, 1], s=5, label="generated / target")

    plt.axis("equal")
    plt.title(title)
    plt.legend()
    save_or_show(title, save_dir=save_dir, show=show)


def plot_trajectory(trajectory, title, max_lines=300, save_dir=None, show=True):
    trajectory = trajectory.detach().cpu()
    n = min(trajectory.shape[1], max_lines)

    plt.figure(figsize=(5, 5))

    for i in range(n):
        path = trajectory[:, i]
        plt.plot(path[:, 0], path[:, 1], alpha=0.18, linewidth=0.8)

    start = trajectory[0, :n]
    end = trajectory[-1, :n]
    plt.scatter(start[:, 0], start[:, 1], s=5, label="noise")
    plt.scatter(end[:, 0], end[:, 1], s=5, label="generated")

    plt.axis("equal")
    plt.title(title)
    plt.legend()
    save_or_show(title, save_dir=save_dir, show=show)


def plot_trajectory_snapshots(
    trajectory,
    title,
    num_snapshots=6,
    max_points=2000,
    save_dir=None,
    show=True,
):
    trajectory = trajectory.detach().cpu()
    indices = torch.linspace(0, trajectory.shape[0] - 1, num_snapshots).long()
    n = min(trajectory.shape[1], max_points)

    fig, axes = plt.subplots(1, num_snapshots, figsize=(3 * num_snapshots, 3))
    if num_snapshots == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        x_t = trajectory[idx, :n]
        t = idx.item() / (trajectory.shape[0] - 1)
        ax.scatter(x_t[:, 0], x_t[:, 1], s=4)
        ax.set_aspect("equal")
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
        choices=["8gaussians", "swissroll", "twomoons"],
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
    z, x_gt = sample_pair(n, dataset, device=device)
    if args.gt_coupling == "optimal":
        x_gt_coupled = optimal_coupling(z, x_gt)
    else:
        x_gt_coupled = x_gt
    x_oracle, traj_oracle = sample_oracle_transport(z, x_gt_coupled, steps=100)

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

    plot_samples(x_gt, "Ground Truth Samples", save_dir=args.save_dir, show=show)
    plot_samples(x_oracle, "Oracle Transport Samples", save_dir=args.save_dir, show=show)
    plot_samples(x_fm, "Flow Matching Samples", save_dir=args.save_dir, show=show)
    plot_samples(x_rf, "Rectified Flow Samples", save_dir=args.save_dir, show=show)
    plot_samples(x_mf, "MeanFlow Samples", save_dir=args.save_dir, show=show)
    plot_samples(x_mf_10, "MeanFlow 10-step Samples", save_dir=args.save_dir, show=show)
    plot_samples(x_mf_50, "MeanFlow 50-step Samples", save_dir=args.save_dir, show=show)

    plot_coupling(z, x_gt_coupled, "Ground Truth Coupling", save_dir=args.save_dir, show=show)
    plot_trajectory(traj_oracle, "Oracle Transport Trajectory", save_dir=args.save_dir, show=show)
    plot_trajectory_snapshots(
        traj_oracle,
        "Oracle Transport Snapshots",
        save_dir=args.save_dir,
        show=show,
    )
    plot_trajectory(traj_fm, "Flow Matching Trajectory", save_dir=args.save_dir, show=show)
    plot_trajectory(traj_rf, "Rectified Flow Trajectory", save_dir=args.save_dir, show=show)
    plot_trajectory(traj_mf, "MeanFlow Trajectory", save_dir=args.save_dir, show=show)
    plot_trajectory(traj_mf_10, "MeanFlow 10-step Trajectory", save_dir=args.save_dir, show=show)
    plot_trajectory(traj_mf_50, "MeanFlow 50-step Trajectory", save_dir=args.save_dir, show=show)

    print("FM:", coupling_metrics(z, x_gt_coupled, x_fm))
    print("RF:", coupling_metrics(z, x_gt_coupled, x_rf))
    print("MeanFlow 1-step:", coupling_metrics(z, x_gt_coupled, x_mf))
    print("MeanFlow 10-step:", coupling_metrics(z, x_gt_coupled, x_mf_10))
    print("MeanFlow 50-step:", coupling_metrics(z, x_gt_coupled, x_mf_50))
