from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import Metric, MaxMetric

from src.metrics.top_k_sim_accuracy import TopKSimAccuracy


class SimCLRModule(LightningModule):
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
        gpu_train_transform = None,
        linear_eval_cfg: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion', 'losses', 'metrics'])

        self.net = net.to(memory_format=torch.channels_last)

        self.criterion = criterion
        self.losses = losses
        self.metrics = metrics

        self.val_acc_best = MaxMetric()

        self.linear_eval_head = None
        self.linear_eval_optimizer = None
        self.linear_eval_criterion = linear_eval_cfg.criterion
        self.linear_eval_losses = linear_eval_cfg.losses
        self.linear_eval_metrics = linear_eval_cfg.metrics

        self.current_dataloader_idx = None

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.losses['val_mode'].reset()
        for _, metric in self.metrics['val_mode'].items():
            metric.reset()
        self.val_acc_best.reset()

    def forward(self, x):
        return self.net(x.to(memory_format=torch.channels_last))

    def get_sim_matrix(self, feats):
        sim = F.cosine_similarity(feats.unsqueeze(0), feats.unsqueeze(1), dim=-1)
        mask = torch.eye(sim.size(0), device=sim.device, dtype=bool)
        masked_sim = torch.where(mask, -torch.inf, sim)
        return masked_sim

    def reset_linear_eval_head(self, device) -> None:
        self.linear_eval_head = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.hparams.feat_dim, self.hparams.num_classes),
        ).to(memory_format=torch.channels_last, device=device)
        self.linear_eval_optimizer = torch.optim.SGD(
            self.linear_eval_head.parameters(), 0.01, momentum=0.9, weight_decay=1e-4
        )

    def linear_eval_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> torch.tensor:
        x, y = batch
        with torch.no_grad():
            feats = self.forward(x)
        logits = self.linear_eval_head(feats)
        loss = self.linear_eval_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        if mode == 'train':
            loss.backward()
            self.linear_eval_optimizer.step()
            self.linear_eval_optimizer.zero_grad()

        self.linear_eval_losses[f'{mode}_mode'](loss)
        self.log(f"linear_eval_{mode}/loss", self.linear_eval_losses[f'{mode}_mode'], on_step=False, on_epoch=True, prog_bar=True)

        for metric_name, metric in self.linear_eval_metrics[f'{mode}_mode'].items():
            metric(preds, y)
            self.log(f"linear_eval_{mode}/{metric_name}", metric, on_step=False, on_epoch=True, prog_bar=True)

        return None

    def unsupervised_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> torch.tensor:
        (imgs1, imgs2), _ = batch
        x = torch.cat([imgs1, imgs2], dim=0)
        if self.hparams.gpu_train_transform is not None:
            x = self.hparams.gpu_train_transform(x)

        feats = self.forward(x)
        loss = self.criterion(feats)

        self.losses[f'{mode}_mode'](loss)
        self.log(f"{mode}/loss", self.losses[f'{mode}_mode'], on_step=False, on_epoch=True, prog_bar=True)

        for metric_name, metric in self.metrics[f'{mode}_mode'].items():
            metric(feats)
            self.log(f"{mode}/{metric_name}", metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> torch.tensor:
        self.current_dataloader_idx = 0

        if self.trainer.datamodule.hparams.linear_eval_datamodule is not None:
            batch, batch_idx, dataloader_idx = batch
            self.current_dataloader_idx = dataloader_idx

            if dataloader_idx == 1:
                if batch_idx == 0 and mode == 'train':
                    self.reset_linear_eval_head(batch[0].device)
                return self.linear_eval_model_step(batch, mode)

        return self.unsupervised_model_step(batch, mode)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.model_step(batch, 'train')

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.model_step(batch, 'val')

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.model_step(batch, 'test')

    def on_validation_epoch_end(self) -> None:
        acc = self.metrics['val_mode']['acc_top1'].compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/best_top1", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.hparams.compile:
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
