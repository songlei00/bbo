import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from bbo.algorithms.surrogates.gp import kernel_impl


def squareplus(X: torch.Tensor):
    return 0.5 * (X + torch.sqrt(X**2 + 4))


class TemplateKernel(nn.Module):
    def __init__(self, impl: kernel_impl.KernelProtocol):
        super(TemplateKernel, self).__init__()
        self.lengthscale = nn.Parameter(torch.randn(()))
        self.transform = F.softplus
        self.impl = impl

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        lengthscale = self.transform(self.lengthscale)
        return self.impl(X1, X2, lengthscale)
    

RBFKernel = functools.partial(TemplateKernel, impl=kernel_impl.rbf_vmap_2d)
Matern52Kernel = functools.partial(TemplateKernel, impl=kernel_impl.matern52_vmap_2d)


class ScaleKernel(nn.Module):
    def __init__(self, base_kernel: nn.Module):
        super(ScaleKernel, self).__init__()
        self.base_kernel = base_kernel
        self.variance = nn.Parameter(torch.randn(()))
        self.transform = F.softplus

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        variance = self.transform(self.variance)
        return variance * self.base_kernel(X1, X2)