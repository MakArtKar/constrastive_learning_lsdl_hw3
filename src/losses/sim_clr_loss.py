import torch
import torch.nn.functional as F


class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats: torch.tensor) -> torch.tensor:
        sim = F.cosine_similarity(feats.unsqueeze(0), feats.unsqueeze(1), dim=-1) / self.temperature
        masked_sim = torch.where(torch.eye(sim.shape[0], device=sim.device, dtype=bool), -torch.inf, sim)
        loss = torch.logsumexp(masked_sim, dim=-1) - torch.roll(masked_sim, feats.size(0) // 2, 0).diag()
        return loss.mean()
