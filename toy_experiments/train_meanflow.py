import torch
import torch.nn.functional as F

from dataset import sample_pair
from model import MeanFlowMLP


def train_meanflow(
    dataset="8gaussians",
    steps=20000,
    batch_size=512,
    lr=1e-3,
    device="cuda",
):
    model = MeanFlowMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        x0, x1 = sample_pair(batch_size, dataset, device=device)

        r = torch.rand(batch_size, 1, device=device)
        t = torch.rand(batch_size, 1, device=device)

        t0 = torch.minimum(r, t)
        t1 = torch.maximum(r, t)

        x_t0 = (1.0 - t0) * x0 + t0 * x1
        x_t1 = (1.0 - t1) * x0 + t1 * x1

        target_avg_v = (x_t1 - x_t0) / (t1 - t0 + 1e-5)

        pred_avg_v = model(x_t0, t0, t1)
        loss = F.mse_loss(pred_avg_v, target_avg_v)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(f"[MeanFlow] step {step:05d} | loss {loss.item():.6f}")

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_meanflow(
        dataset="8gaussians",
        steps=20000,
        batch_size=512,
        device=device,
    )

    torch.save(model.state_dict(), "meanflow_8gaussians.pt")