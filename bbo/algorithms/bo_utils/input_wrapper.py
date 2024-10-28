from typing import List

import torch 
from torch import nn 
from gpytorch.means import Mean
from gpytorch.kernels import Kernel


class Squareplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x**2 + 4))


class IdentityWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MLPWrapper(nn.Module):
    def __init__(
        self,
        hidden_features: List,
        activate_final: bool=False,
    ):
        super().__init__()
        if len(hidden_features) < 2:
            raise ValueError('hidden_features must have at least 2 feature')
        mlp = []
        in_features = hidden_features[0]
        for i in hidden_features[1:-1]:
            mlp.append(nn.Linear(in_features, i))
            mlp.append(nn.Tanh())
            in_features = i
        mlp.append(nn.Linear(in_features, hidden_features[-1]))
        if activate_final:
            mlp.append(nn.Tanh())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class KumarWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Softplus()
        # self.transform = Squareplus()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = 1e-6

    def forward(self, x):
        x = x.clip(self.eps, 1-self.eps)
        alpha = self.transform(self.alpha)
        beta = self.transform(self.beta)

        res = 1 - (1 - x.pow(alpha)).pow(beta)
        return res


class WrapperMean(Mean):
    def __init__(
        self,
        wrapper: nn.Module
    ):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, x):
        m = self.wrapper(x)
        return m.squeeze(-1)


class WrapperKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        wrapper: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.wrapper = wrapper

    def forward(self, x1, x2, **params):
        return self.base_kernel(self.wrapper(x1), self.wrapper(x2), **params)