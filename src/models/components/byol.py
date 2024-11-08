from copy import deepcopy

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage


class BYOL(torch.nn.Module):
    def __init__(
        self, backbone: torch.nn.Module, projection_head: torch.nn.Module,
        prediction_head: torch.nn.Module, num_steps: int, momentum: float = 0.996,
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.prediction_head = prediction_head

        self.backbone_ema = None
        self.projection_ema = None
        
        self.num_steps = num_steps
        self.momentum = momentum
        self.k = 0

    def setup(self):
        if self.backbone_ema is None and self.projection_ema is None:
            self.backbone_ema = ExponentialMovingAverage(self.backbone.parameters(), decay=self.momentum)
            self.projection_ema = ExponentialMovingAverage(self.projection_head.parameters(), decay=self.momentum)

    def forward(self, x, use_momentum=False):
        self.setup()
        if use_momentum:
            with self.backbone_ema.average_parameters():
                y = self.backbone(x)
            
            with self.projection_ema.average_parameters():
                z = self.projection_head(y)

            return [y, z]
        else:
            y = self.backbone(x)
            z = self.projection_head(y)
            q = self.prediction_head(z)
            return [y, z, q]

    def update_momentum_net(self):
        self.setup()
        self.backbone_ema.update()
        self.projection_ema.update()

        self.k += 1
        cur_momentum = 1 - (1 - self.momentum) * (np.cos(np.pi * self.k / self.num_steps) + 1) / 2
        self.backbone_ema.decay = cur_momentum
        self.projection_ema.decay = cur_momentum

