from collections import defaultdict
from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.losses.sim_clr_loss import SimCLRLoss


class SimCLRModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        gpu_train_transform = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        net.fc = torch.nn.Sequential(
            net.fc,
            torch.nn.ReLU(),
            torch.nn.Linear(net.fc.out_features, net.fc.out_features // 4)
        )
        self.net = net.to(memory_format=torch.channels_last)
        # self.net = net

        self.criterion = criterion

        self.train_acc_top1 = Accuracy(task="multiclass", num_classes=batch_size * 2, top_k=1)
        self.train_acc_top5 = Accuracy(task="multiclass", num_classes=batch_size * 2, top_k=5)
        self.val_acc_top1 = Accuracy(task="multiclass", num_classes=batch_size * 2, top_k=1)
        self.val_acc_top5 = Accuracy(task="multiclass", num_classes=batch_size * 2, top_k=5)
        self.test_acc_top1 = Accuracy(task="multiclass", num_classes=batch_size * 2, top_k=1)
        self.test_acc_top5 = Accuracy(task="multiclass", num_classes=batch_size * 2, top_k=5)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.net(x.to(memory_format=torch.channels_last))
        # return self.net(x)

    def get_sim_matrix(self, feats):
        sim = F.cosine_similarity(feats.unsqueeze(0), feats.unsqueeze(1), dim=-1)
        mask = torch.eye(sim.size(0), device=sim.device, dtype=bool)
        masked_sim = torch.where(mask, -torch.inf, sim)
        return masked_sim

    def model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor],
    ) -> torch.tensor:
        (imgs1, imgs2), _ = batch
        x = torch.cat([imgs1, imgs2], dim=0)
        if self.hparams.gpu_train_transform is not None:
            x = self.hparams.gpu_train_transform(x)
        feats = self.forward(x)
        loss = self.criterion(feats)

        scores = -self.get_sim_matrix(feats)
        if scores.shape[1] < self.hparams.batch_size * 2:  # last batch
            scores = torch.cat([scores, -torch.ones(scores.shape[0], self.hparams.batch_size * 2 - scores.shape[1], device=scores.device) * torch.inf], dim=1)
        assert scores.shape[1] == self.hparams.batch_size * 2
        gts = torch.arange(scores.shape[0], dtype=int, device=scores.device).roll(self.hparams.batch_size, dims=0)
        return loss, scores, gts

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, scores, gts = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc_top1(scores, gts)
        self.log(f"train/acc_top1", self.train_acc_top1, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc_top5(scores, gts)
        self.log(f"train/acc_top5", self.train_acc_top5, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, scores, gts = self.model_step(batch)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_acc_top1(scores, gts)
        self.log(f"val/acc_top1", self.val_acc_top1, on_step=False, on_epoch=True, prog_bar=True)

        self.val_acc_top5(scores, gts)
        self.log(f"val/acc_top5", self.val_acc_top5, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, scores, gts = self.model_step(batch)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_acc_top1(scores, gts)
        self.log(f"test/acc_top1", self.test_acc_top1, on_step=False, on_epoch=True, prog_bar=True)

        self.test_acc_top5(scores, gts)
        self.log(f"test/acc_top5", self.test_acc_top5, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc_top1.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/best_top1", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if stage == "fit" and self.hparams.compile:
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        self.hparams.optimizer.keywords['lr'] = self.hparams.optimizer.keywords['lr'] * self.hparams.batch_size / 256
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
