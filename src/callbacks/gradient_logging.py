import lightning as pl
import torch


class GradientLoggingCallback(pl.Callback):
    def __init__(self, grad_clip_val = 1.):
        self.grad_clip_val = grad_clip_val

    def on_after_backward(self, trainer, pl_module):
        # Manually log the gradient norm
        torch.nn.utils.clip_grad_norm_(pl_module.parameters(), self.grad_clip_val)

        grad_norms = [p.grad.norm(2).item() for p in pl_module.parameters() if p.grad is not None]
        mean_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0

        trainer.logger.log_metrics({"mean_grad_norm": mean_grad_norm}, step=trainer.global_step)
