from typing import Optional

import lightning as pl
import torch


class GradientLoggingCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        grad_norms = [p.grad.norm(2).item() for p in pl_module.parameters() if p.grad is not None]
        mean_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0

        trainer.logger.log_metrics({"mean_grad_norm": mean_grad_norm}, step=trainer.global_step)
