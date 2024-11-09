from abc import abstractmethod
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import Metric, MaxMetric

from src.models.base_module import BaseModule
from src.models.components.base import EncoderWithHead


class LinearEvalModule(BaseModule):
    def __init__(
        self,
        net: EncoderWithHead,
        feat_dim: int,
        num_classes: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        losses: Metric,
        metrics: Metric,
        compile: bool,
        linear_eval_cfg: Dict[str, Any],
        val_best_name: str = 'acc',
    ) -> None:
        super().__init__(
            net, criterion, optimizer, scheduler, losses, metrics, compile,
            ignore=['net', 'criterion', 'losses', 'metrics'], val_best_name=val_best_name,
        )

        self.linear_eval_head = None
        self.linear_eval_optimizer = None
        self.linear_eval_criterion = linear_eval_cfg.criterion
        self.linear_eval_losses = linear_eval_cfg.losses
        self.linear_eval_metrics = linear_eval_cfg.metrics

        self.current_dataloader_idx = None
        self.reset_linear_eval_head('cpu')

    @abstractmethod
    def unsupervised_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Returns tuple of `criterion_inputs` and `metrics_inputs`
            * `criterion_inputs` - *args for self.criterion call
            * `metrics_inputs` - *args for self.metrics call
        Inputs:
            * `batch` - batch of dataloader
            * `mode` - current mode, one of ('train', 'val', 'test')
        """
        raise NotImplementedError()

    @abstractmethod
    def linear_eval_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Returns tuple of `feats` and `targets`
            * `feats` - encoded representation of `batch`
            * `targets` - targets of batch, will be passed to criterion and metrics as *args
        Inputs:
            * `batch` - batch of dataloader
            * `mode` - current mode, one of ('train', 'val', 'test')
        """
        raise NotImplementedError()

    def reset_linear_eval_head(self, device) -> None:
        self.linear_eval_head = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.hparams.feat_dim, self.hparams.num_classes),
        ).to(memory_format=torch.channels_last, device=device)
        self.linear_eval_optimizer = torch.optim.SGD(
            self.linear_eval_head.parameters(), 0.01, momentum=0.9, weight_decay=1e-4
        )

    def _linear_eval_model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ):
        feats, targets = self.linear_eval_model_step(batch, mode)

        logits = self.linear_eval_head(feats)
        loss = self.linear_eval_criterion(logits, *targets)
        preds = torch.argmax(logits, dim=1)

        if mode == 'train':
            loss.backward()
            self.linear_eval_optimizer.step()
            self.linear_eval_optimizer.zero_grad()

        self.logging_step(
            mode, loss, preds, *targets, prefix="linear_eval_",
            losses=self.linear_eval_losses, metrics=self.linear_eval_metrics
        )
        return None

    def model_step(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.Tensor], mode: str
    ) -> torch.tensor:

        batch, batch_idx, self.current_dataloader_idx = batch

        if self.current_dataloader_idx == 0:
            return self.unsupervised_model_step(batch, mode)

        if batch_idx == 0 and mode == 'train':
            self.reset_linear_eval_head(batch[0].device)

        return self._linear_eval_model_step(batch, mode)
