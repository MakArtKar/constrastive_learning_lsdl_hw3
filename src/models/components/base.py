from typing import Any

import torch


class EncoderWithHead(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: Any, return_list: bool = False) -> Any:
        feats = self.forward_encoder(x)
        out = self.head(feats)
        if return_list:
            return [feats, out]
        else:
            return out

    def forward_encoder(self, x: Any) -> Any:
        return self.encoder(x)
