from typing import Any, Dict, Tuple

import torch
from torchmetrics import Metric, MaxMetric

from src.models.components.byol import BYOL
from src.models.linear_eval_base_module import LinearEvalModule


class BYOLModule(LinearEvalModule):
    def __init__(
        self,
        net: BYOL,
        feat_dim: int,
        num_classes: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        losses: Metric,
        metrics: Metric,
        compile: bool,
        linear_eval_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(
            net, feat_dim, num_classes, criterion, optimizer, scheduler,
            losses, metrics, compile, linear_eval_cfg, val_best_name='acc_top1',
        )

    def log_z_std(self, z: torch.tensor, key: str) -> None:
        std = torch.std(z, dim=0).mean()
        self.log(key, std.item(), sync_dist=True, prog_bar=True)

    def unsupervised_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> torch.tensor:
        (imgs1, imgs2), _ = batch
        x = torch.cat([imgs1, imgs2], dim=0)
        y, z, q = self.net.forward(x, return_list=True)
        with torch.no_grad():
            y_ema, z_ema = self.net.forward(x, use_momentum=True, return_list=True)
        q1, q2 = q[:imgs1.size(0)], q[imgs1.size(0):]
        z_ema1, z_ema2 = z_ema[:imgs1.size(0)], z_ema[imgs1.size(0):]
        self.log_z_std(z, 'z_std')
        self.log_z_std(z_ema, 'z_ema_std')
        return [q1, z_ema2, q2, z_ema1], [y, y_ema]

    def linear_eval_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> Tuple[torch.tensor, torch.tensor]:
        x, y = batch
        with torch.no_grad():
            feats = self.net.encoder(x)
        return feats, [y]

    def on_before_zero_grad(self, optimizer):
        self.net.update_momentum_net()

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.linear_eval_metrics['val_mode']['acc'].compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(f"linear_eval_val/best_acc", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
