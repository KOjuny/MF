import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import sample_pair
from model import FlowMatchingMLP


def train_rectified_flow(
    dataset="8gaussians",
    steps=20000,
    batch_size=512,
    lr=1e-3,
    device="cuda",
):
    model = FlowMatchingMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        x0, x1 = sample_pair(batch_size, dataset, device=device)

        t = torch.rand(batch_size, 1, device=device)

        # Rectified Flow linear interpolation
        x_t = (1.0 - t) * x0 + t * x1

        # Rectified Flow target velocity
        target_v = x1 - x0

        pred_v = model(x_t, t)
        loss = F.mse_loss(pred_v, target_v)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(f"[Rectified Flow] step {step:05d} | loss {loss.item():.6f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="8gaussians")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=Path, default=Path("./results/models"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_rectified_flow(
        dataset=args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_dir / f"rectified_flow_{args.dataset}.pt")
