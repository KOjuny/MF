import torch
import matplotlib.pyplot as plt

from workspace.MeanFlow.MF.toy_experiments.dataset import sample_pair
from model import FlowMatchingMLP, MeanFlowMLP


@torch.no_grad()
def sample_flow_matching(model, z, steps=100):
    x = z.clone()
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((z.shape[0], 1), i / steps, device=z.device)
        v = model(x, t)
        x = x + dt * v

    return x


@torch.no_grad()
def sample_meanflow(model, z):
    r = torch.zeros(z.shape[0], 1, device=z.device)
    t = torch.ones(z.shape[0], 1, device=z.device)

    avg_v = model(z, r, t)
    x = z + avg_v
    return x

@torch.no_grad()
def sample_meanflow_multistep(model, z, steps=10):
    x = z.clone()

    for i in range(steps):
        r = torch.full((z.shape[0], 1), i / steps, device=z.device)
        t = torch.full((z.shape[0], 1), (i + 1) / steps, device=z.device)

        avg_v = model(x, r, t)
        x = x + (t - r) * avg_v

    return x


def plot_samples(x, title):
    x = x.detach().cpu()

    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=4)
    plt.axis("equal")
    plt.title(title)
    plt.show()


def plot_coupling(z, x, title, max_lines=300):
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
    plt.show()


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "8gaussians"

    n = 2000
    z, x_gt = sample_pair(n, dataset, device=device)

    fm = FlowMatchingMLP().to(device)
    fm.load_state_dict(torch.load("flow_matching_8gaussians.pt", map_location=device))
    fm.eval()

    rf = FlowMatchingMLP().to(device)
    rf.load_state_dict(torch.load("rectified_flow_8gaussians.pt", map_location=device))
    rf.eval()

    mf = MeanFlowMLP().to(device)
    mf.load_state_dict(torch.load("meanflow_8gaussians.pt", map_location=device))
    mf.eval()


    x_fm = sample_flow_matching(fm, z, steps=100)
    x_rf = sample_flow_matching(rf, z, steps=100)
    x_mf = sample_meanflow(mf, z)
    x_mf_10 = sample_meanflow_multistep(mf, z, steps=10)
    x_mf_50 = sample_meanflow_multistep(mf, z, steps=50)

    plot_samples(x_gt, "Ground Truth Samples")
    plot_samples(x_fm, "Flow Matching Samples")
    plot_samples(x_rf, "Rectified Flow Samples")
    plot_samples(x_mf, "MeanFlow Samples")
    plot_samples(x_mf_10, "MeanFlow 10-step Samples")
    plot_samples(x_mf_50, "MeanFlow 50-step Samples")

    plot_coupling(z, x_gt, "Ground Truth Coupling")
    plot_coupling(z, x_fm, "Flow Matching Coupling")
    plot_coupling(z, x_rf, "Rectified Flow Coupling")
    plot_coupling(z, x_mf, "MeanFlow Coupling")
    plot_coupling(z, x_mf_10, "MeanFlow 10-step Coupling")
    plot_coupling(z, x_mf_50, "MeanFlow 50-step Coupling")

    print("FM:", coupling_metrics(z, x_gt, x_fm))
    print("RF:", coupling_metrics(z, x_gt, x_rf))
    print("MeanFlow 1-step:", coupling_metrics(z, x_gt, x_mf))
    print("MeanFlow 10-step:", coupling_metrics(z, x_gt, x_mf_10))
    print("MeanFlow 50-step:", coupling_metrics(z, x_gt, x_mf_50))