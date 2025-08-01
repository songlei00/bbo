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

import torch
import pytest

from bbo.algorithms.surrogates.gp import kernels, kernel_impl, warpers

bs, n1, n2, d = 4, 20, 10, 3


@pytest.mark.parametrize("kernel", [
    # Test basic kernel
    kernels.RBFKernel(),
    kernels.Matern52Kernel(),
    kernels.TemplateKernel(kernel_impl.rbf_cdist),
    kernels.TemplateKernel(kernel_impl.matern52_cdist),
    # Test scale kernel
    kernels.ScaleKernel(kernels.RBFKernel()),
    # Test warp kernel
    kernels.WarpKernel(
        base_kernel=kernels.ScaleKernel(kernels.RBFKernel()),
        warp_module=warpers.KumarWarp()
    ),
    kernels.WarpKernel(
        base_kernel=kernels.ScaleKernel(kernels.RBFKernel(ard_dim=10)),
        warp_module=warpers.MLPWarp(d, [32,], 10)
    )
])
class TestKernelRun:
    def test_kernel_2d(self, kernel):
        X1 = torch.randn(n1, d)
        X2 = torch.randn(n2, d)
        K = kernel(X1, X2)
        assert K.shape == (n1, n2)

    def test_kernel_3d(self, kernel):
        X1 = torch.randn(bs, n1, d)
        X2 = torch.randn(bs, n2, d)
        K = kernel(X1, X2)
        assert K.shape == (bs, n1, n2)


def test_parameter():
    kernel = kernels.ScaleKernel(kernels.RBFKernel())
    assert len(list(kernel.parameters())) == 2