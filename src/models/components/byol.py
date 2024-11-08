from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from src.models.components.base import EncoderWithHead


class BYOL(EncoderWithHead):
    def __init__(
        self, encoder: torch.nn.Module, projection_head: torch.nn.Module,
        prediction_head: torch.nn.Module, num_steps: int, momentum: float = 0.996,
    ):
        super().__init__(
            encoder,
            torch.nn.ModuleDict({
                'projection_head': projection_head,
                'prediction_head': prediction_head,
            }),
        )
        self.encoder_ema = None
        self.projection_ema = None
        
        self.num_steps = num_steps
        self.momentum = momentum
        self.k = 0

    def setup(self):
        if self.encoder_ema is None and self.projection_ema is None:
            self.encoder_ema = ExponentialMovingAverage(self.encoder.parameters(), decay=self.momentum)
            self.projection_ema = ExponentialMovingAverage(self.head.projection_head.parameters(), decay=self.momentum)

    def forward(self, x, return_list=False, use_momentum=False):
        y = self.forward_encoder(x, use_momentum=use_momentum)
        if use_momentum:
            with self.projection_ema.average_parameters():
                z = self.head.projection_head(y)

            if return_list:
                return [y, z]
            else:
                return z
        else:
            z = self.head.projection_head(y)
            q = self.head.prediction_head(z)
            if return_list:
                return [y, z, q]
            else:
                return q

    def forward_encoder(self, x: Any, use_momentum=False) -> Any:
        if use_momentum:
            self.setup()
            with self.encoder_ema.average_parameters():
                y = self.encoder(x)
            return y
        else:
            return self.encoder(x)

    def update_momentum_net(self):
        self.setup()
        self.encoder_ema.update()
        self.projection_ema.update()

        self.k += 1
        cur_momentum = 1 - (1 - self.momentum) * (np.cos(np.pi * self.k / self.num_steps) + 1) / 2
        self.encoder_ema.decay = cur_momentum
        self.projection_ema.decay = cur_momentum
