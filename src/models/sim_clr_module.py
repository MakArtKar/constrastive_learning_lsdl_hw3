from typing import Any, Dict, Tuple

import torch
from torchmetrics import Metric, MaxMetric

from src.models.linear_eval_base_module import LinearEvalModule


class SimCLRModule(LinearEvalModule):
    def __init__(
        self,
        net: torch.nn.Module,
        feat_dim: int,
        num_classes: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        losses: Metric,
        metrics: Metric,
        compile: bool,
        linear_eval_cfg: Dict[str, Any],
        gpu_train_transform = None,
    ) -> None:
        super().__init__(
            net, feat_dim, num_classes, criterion, optimizer, scheduler,
            losses, metrics, compile, linear_eval_cfg, val_best_name='acc_top1',
        )
        self.gpu_train_transform = gpu_train_transform

    def unsupervised_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> Any:
        (imgs1, imgs2), _ = batch
        x = torch.cat([imgs1, imgs2], dim=0)
        if self.hparams.gpu_train_transform is not None:
            x = self.hparams.gpu_train_transform(x)
        feats = self.forward(x)
        return [feats], [feats, feats]

    def linear_eval_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> Any:
        x, y = batch
        with torch.no_grad():
            feats = self.forward(x)
        return feats, [y]
