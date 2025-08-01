# Copyright 2025 songlei
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from bbo.algorithms.surrogates.gp import kernel_impl


def squareplus(X: torch.Tensor):
    return 0.5 * (X + torch.sqrt(X**2 + 4))


class TemplateKernel(nn.Module):
    def __init__(self, impl: kernel_impl.KernelProtocol, ard_dim: int | None = None):
        super(TemplateKernel, self).__init__()
        if ard_dim is not None:
            self.lengthscale = nn.Parameter(torch.randn(ard_dim))
        else:
            self.lengthscale = nn.Parameter(torch.randn(()))
        self.transform = F.softplus
        self.impl = impl

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        lengthscale = self.transform(self.lengthscale)
        return self.impl(X1, X2, lengthscale)
    

RBFKernel = functools.partial(TemplateKernel, impl=kernel_impl.rbf_vmap)
Matern52Kernel = functools.partial(TemplateKernel, impl=kernel_impl.matern52_vmap)


class ScaleKernel(nn.Module):
    def __init__(self, base_kernel: nn.Module):
        super(ScaleKernel, self).__init__()
        self.base_kernel = base_kernel
        self.variance = nn.Parameter(torch.randn(()))
        self.transform = F.softplus

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        variance = self.transform(self.variance)
        return variance * self.base_kernel(X1, X2)
    

class WarpKernel(nn.Module):
    def __init__(self, base_kernel: nn.Module, warp_module: nn.Module):
        super(WarpKernel, self).__init__()
        self.warp_module = warp_module
        self.base_kernel = base_kernel

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        X1 = self.warp_module(X1)
        X2 = self.warp_module(X2)
        return self.base_kernel(X1, X2)
