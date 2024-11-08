from copy import deepcopy

import torch
from torch_ema import ExponentialMovingAverage


class BYOL(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, online_head: torch.nn.Module, momentum: float):
        super().__init__()
        self.backbone = backbone
        self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=momentum)
        self.online_head = online_head

    def forward(self, x, use_momentum=False):
        if use_momentum:
            with self.ema.average_parameters():
                projection = self.backbone(x)
        else:
            projection = self.backbone(x)
        return self.online_head(projection)

    def update_momentum_backbone(self):
        self.ema.update()
