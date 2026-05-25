import torch
import torch.nn.functional as F

from dataset import sample_pair
from model import FlowMatchingMLP


def train_flow_matching(
    dataset="8gaussians",
    steps=20000,
    batch_size=512,
    lr=1e-3,
    sigma_min=0.01,
    device="cuda",
):
    model = FlowMatchingMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        _, x1 = sample_pair(batch_size, dataset, device=device)

        t = torch.rand(batch_size, 1, device=device)
        eps = torch.randn_like(x1)

        sigma_t = 1.0 - (1.0 - sigma_min) * t

        x_t = t * x1 + sigma_t * eps

        target_v = (x1 - (1.0 - sigma_min) * x_t) / sigma_t

        pred_v = model(x_t, t)
        loss = F.mse_loss(pred_v, target_v)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(f"[Flow Matching] step {step:05d} | loss {loss.item():.6f}")

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_flow_matching(
        dataset="8gaussians",
        steps=20000,
        batch_size=512,
        device=device,
    )

    torch.save(model.state_dict(), "flow_matching_8gaussians.pt")