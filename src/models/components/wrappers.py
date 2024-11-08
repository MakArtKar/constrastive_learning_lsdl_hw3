from typing import Dict

import torch


class ModelConstructor(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, modules: Dict[str, torch.nn.Module]):
        super().__init__()
        self.net = net
        for module_name, module in modules.items():
            setattr(self.net, module_name, module)

    def forward(self, x):
        return self.net(x)
