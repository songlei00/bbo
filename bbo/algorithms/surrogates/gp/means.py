import torch
import torch.nn as nn


class ConstantMean(nn.Module):
    def __init__(self):
        super(ConstantMean, self).__init__()
        self.constant = nn.Parameter(torch.zeros(()))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        return self.constant.expand(n, 1)