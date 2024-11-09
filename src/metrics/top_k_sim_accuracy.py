import torch
from torchmetrics import Metric
import torch.nn.functional as F


class TopKSimAccuracy(Metric):
    def __init__(self, top_k: int = 1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k 
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, feats1: torch.Tensor, feats2: torch.Tensor):
        batch_size, feat_dim = feats1.size()

        similarity_matrix = F.cosine_similarity(feats1.unsqueeze(0), feats2.unsqueeze(1), dim=-1)

        _, topk_indices = similarity_matrix.topk(self.top_k, dim=1)
        pos_pairs = torch.arange(feats1.size(0), device=feats1.device)

        is_correct = (topk_indices == pos_pairs.unsqueeze(1)).any(dim=1)
        self.correct += is_correct.sum()
        self.total += batch_size
        
    def compute(self):
        return self.correct.float() / self.total
