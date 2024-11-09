from abc import abstractmethod
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import Metric, MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class BaseModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        losses: Metric,
        metrics: Metric,
        compile: bool,
        ignore: list = None,
        val_best_name: str = 'acc',
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        ignore = ignore or ['net', 'losses', 'metrics']
        self.save_hyperparameters(logger=False, ignore=ignore)

        self.net = net.to(memory_format=torch.channels_last)

        # loss function
        self.criterion = criterion

        self.losses = losses
        self.metrics = metrics

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    @abstractmethod
    def model_step(
        self, batch: Any, mode: str
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

    def _model_step(
        self, batch: Any, mode: str
    ) -> torch.Tensor:
        inputs = self.model_step(batch, mode)
        if inputs is None:
            return None
        criterion_inputs, metrics_inputs = inputs
        loss = self.criterion(*criterion_inputs)
        self.logging_step(mode, loss, *metrics_inputs)
        return loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.losses['val_mode'].reset()
        for _, metric in self.metrics['val_mode'].items():
            metric.reset()
        self.val_acc_best.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x.to(memory_format=torch.channels_last))

    def logging_step(
        self, mode: str, loss: torch.Tensor, *metric_args,
        prefix="", losses=None, metrics=None
    ):
        losses = losses or self.losses
        metrics = metrics or self.metrics

        losses[f'{mode}_mode'](loss)
        self.log(f"{prefix}{mode}/loss", losses[f'{mode}_mode'], on_step=False, on_epoch=True, prog_bar=True)

        if metrics is not None:
            for metric_name, metric in metrics[f'{mode}_mode'].items():
                metric(*metric_args)
                self.log(f"{prefix}{mode}/{metric_name}", metric, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        return self._model_step(batch, 'train')

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        return self._model_step(batch, 'val')

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.metrics['val_mode'][self.hparams.val_best_name].compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(f"val/best_{self.hparams.val_best_name}", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        return self._model_step(batch, 'test')

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
