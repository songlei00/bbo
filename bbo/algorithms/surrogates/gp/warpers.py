from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_d: int,
        out_d: int,
        norm: Optional[nn.Module] = nn.BatchNorm1d,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        mlp = [nn.Linear(in_d, out_d, bias)]
        if norm is not None:
            mlp.append(norm(out_d))
        mlp.append(activation())
        if dropout > 0:
            mlp.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mlp(X)


class MLPWarp(nn.Module):
    def __init__(
        self,
        in_d: int,
        hidden_d: List[int],
        out_d: int,
        norm: Optional[nn.Module] = None,
        activation: nn.Module = nn.Tanh,
        dropout: float = 0.0,
        bias: bool = True,
        activation_out: Optional[nn.Module] = None
    ):
        super(MLPWarp, self).__init__()
        layers = []
        for i in range(len(hidden_d)):
            layers.append(MLPBlock(in_d, hidden_d[i], norm, activation, dropout, bias))
            in_d = hidden_d[i]
        layers.append(nn.Linear(in_d, out_d, bias))
        if activation_out is not None:
            layers.append(activation_out())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


class KumarWarp(nn.Module):
    def __init__(self):
        super(KumarWarp, self).__init__()
        self.transform = F.softplus
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = 1e-6

    def forward(self, X: torch.Tensor):
        X = X.clip(self.eps, 1-self.eps)
        alpha = self.transform(self.alpha)
        beta = self.transform(self.beta)
        res = 1 - (1 - X.pow(alpha)).pow(beta)
        return res
