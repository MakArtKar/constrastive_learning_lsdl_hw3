from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.metrics.top_k_sim_accuracy import TopKSimAccuracy


class SimCLRModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        gpu_train_transform = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net.to(memory_format=torch.channels_last)

        self.criterion = criterion

        self.setup_metrics()
        self.setup_losses()
        self.val_acc_best = MaxMetric()

    def setup_metrics(self):
        metrics_dict = {}
        for mode in ('train', 'val', 'test'):
            metrics_dict[f'{mode}_mode'] = torch.nn.ModuleDict({
                'acc_top1': TopKSimAccuracy(top_k=1),
                'acc_top5': TopKSimAccuracy(top_k=5),
            })
        self.metrics = torch.nn.ModuleDict(metrics_dict)

    def setup_losses(self):
        self.losses = torch.nn.ModuleDict({
            'train_mode': MeanMetric(),
            'val_mode': MeanMetric(),
            'test_mode': MeanMetric(),
        })

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

    def model_step(
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

        return loss, feats

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, feats = self.model_step(batch, 'train')
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, feats = self.model_step(batch, 'val')
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, feats = self.model_step(batch, 'test')
        return loss

    def on_validation_epoch_end(self) -> None:
        acc = self.metrics['val_mode']['acc_top1'].compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/best_top1", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.hparams.compile:
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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

