import torch
from torchmetrics import Metric
import torch.nn.functional as F


class TopKSimAccuracy(Metric):
    def __init__(self, top_k: int = 1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k 
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, feats: torch.Tensor):
        batch_size, feat_dim = feats.size()
        assert batch_size % 2 == 0

        similarity_matrix = F.cosine_similarity(feats.unsqueeze(0), feats.unsqueeze(1), dim=-1)

        mask = torch.eye(batch_size, dtype=torch.bool, device=feats.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -torch.inf)

        pos_pairs = torch.arange(batch_size, device=feats.device).roll(batch_size // 2, dims=0)

        _, topk_indices = similarity_matrix.topk(self.top_k, dim=1)

        is_correct = (topk_indices == pos_pairs.unsqueeze(1)).any(dim=1)
        self.correct += is_correct.sum()
        self.total += batch_size
        
    def compute(self):
        return self.correct.float() / self.total
