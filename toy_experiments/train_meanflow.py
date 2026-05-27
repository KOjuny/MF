import argparse
from pathlib import Path

import torch
from torch.func import jvp

from dataset import sample_pair
from model import MeanFlowMLP


def sample_meanflow_times(
    batch_size,
    device,
    time_sampler="logit_normal",
    time_mu=-0.4,
    time_sigma=1.0,
    ratio_r_not_equal_t=0.25,
):
    if time_sampler == "uniform":
        time_samples = torch.rand(batch_size, 2, device=device)
    elif time_sampler == "logit_normal":
        time_samples = torch.randn(batch_size, 2, device=device)
        time_samples = torch.sigmoid(time_samples * time_sigma + time_mu)
    else:
        raise ValueError(f"Unknown time sampler: {time_sampler}")

    time_samples, _ = torch.sort(time_samples, dim=1)
    r = time_samples[:, 0:1]
    t = time_samples[:, 1:2]

    fraction_equal = 1.0 - ratio_r_not_equal_t
    equal_mask = torch.rand(batch_size, 1, device=device) < fraction_equal
    r = torch.where(equal_mask, t, r)

    return r, t


def meanflow_loss(
    model,
    data,
    noise,
    time_sampler="logit_normal",
    time_mu=-0.4,
    time_sigma=1.0,
    ratio_r_not_equal_t=0.25,
    weighting="adaptive",
    adaptive_p=1.0,
):
    batch_size = data.shape[0]
    device = data.device

    r, t = sample_meanflow_times(
        batch_size,
        device,
        time_sampler=time_sampler,
        time_mu=time_mu,
        time_sigma=time_sigma,
        ratio_r_not_equal_t=ratio_r_not_equal_t,
    )

    x_t = (1.0 - t) * data + t * noise
    v_t = noise - data

    u = model(x_t, r, t)

    def meanflow_fn(x, r_in, t_in):
        return model(x, r_in, t_in)

    _, dudt = jvp(
        meanflow_fn,
        (x_t, r, t),
        (v_t, torch.zeros_like(r), torch.ones_like(t)),
    )
    u_target = v_t - (t - r) * dudt

    error = u - u_target.detach()
    loss_mid = torch.sum(error.square().reshape(error.shape[0], -1), dim=-1)

    if weighting == "adaptive":
        weights = 1.0 / (loss_mid.detach() + 1e-3).pow(adaptive_p)
        loss = weights * loss_mid
    elif weighting == "uniform":
        loss = loss_mid
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    loss_ref = error.square().mean()
    return loss.mean(), loss_ref


def train_meanflow(
    dataset="8gaussians",
    steps=20000,
    batch_size=512,
    lr=1e-4,
    time_sampler="logit_normal",
    time_mu=-0.4,
    time_sigma=1.0,
    ratio_r_not_equal_t=0.25,
    weighting="adaptive",
    adaptive_p=1.0,
    device="cuda",
):
    model = MeanFlowMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        noise, data = sample_pair(batch_size, dataset, device=device)

        loss, loss_ref = meanflow_loss(
            model,
            data,
            noise,
            time_sampler=time_sampler,
            time_mu=time_mu,
            time_sigma=time_sigma,
            ratio_r_not_equal_t=ratio_r_not_equal_t,
            weighting=weighting,
            adaptive_p=adaptive_p,
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(
                f"[MeanFlow] step {step:05d} | "
                f"loss {loss.item():.6f} | loss_ref {loss_ref.item():.6f}"
            )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="8gaussians")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--time-sampler", default="logit_normal", choices=["uniform", "logit_normal"])
    parser.add_argument("--time-mu", type=float, default=-0.4)
    parser.add_argument("--time-sigma", type=float, default=1.0)
    parser.add_argument("--ratio-r-not-equal-t", type=float, default=0.25)
    parser.add_argument("--weighting", default="adaptive", choices=["uniform", "adaptive"])
    parser.add_argument("--adaptive-p", type=float, default=1.0)
    parser.add_argument("--save-dir", type=Path, default=Path("./results/models"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_meanflow(
        dataset=args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        time_sampler=args.time_sampler,
        time_mu=args.time_mu,
        time_sigma=args.time_sigma,
        ratio_r_not_equal_t=args.ratio_r_not_equal_t,
        weighting=args.weighting,
        adaptive_p=args.adaptive_p,
        device=device,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_dir / f"meanflow_{args.dataset}.pt")
