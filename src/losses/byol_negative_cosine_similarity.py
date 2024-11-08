import torch


class BYOLNegativeCosineSimilarity(torch.nn.CosineSimilarity):
    def forward(self, q1: torch.tensor, z_ema2: torch.tensor, q2: torch.tensor, z_ema1: torch.tensor) -> torch.Tensor:
        return 2 - 2 * torch.mean(super().forward(q1, z_ema2) + super().forward(q2, z_ema1))
