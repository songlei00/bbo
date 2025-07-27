import torch
import pytest

from bbo.algorithms.surrogates.gp import kernels, kernel_impl

bs, n1, n2, d = 4, 20, 10, 3


@pytest.mark.parametrize("kernel", [
    kernels.RBFKernel(),
    kernels.Matern52Kernel(),
    kernels.TemplateKernel(kernel_impl.rbf_cdist),
    kernels.TemplateKernel(kernel_impl.matern52_cdist),
    kernels.ScaleKernel(kernels.RBFKernel())
])
def test_kernel_2d(kernel):
    X1 = torch.randn(n1, d)
    X2 = torch.randn(n2, d)
    K = kernel(X1, X2)
    assert K.shape == (n1, n2)


@pytest.mark.parametrize("kernel", [
    kernels.TemplateKernel(kernel_impl.rbf_vmap_3d),
    kernels.TemplateKernel(kernel_impl.matern52_vmap_3d),
    kernels.TemplateKernel(kernel_impl.rbf_cdist),
    kernels.TemplateKernel(kernel_impl.matern52_cdist),
    kernels.ScaleKernel(kernels.TemplateKernel(kernel_impl.rbf_vmap_3d))
])
def test_kernel_3d(kernel):
    X1 = torch.randn(bs, n1, d)
    X2 = torch.randn(bs, n2, d)
    K = kernel(X1, X2)
    assert K.shape == (bs, n1, n2)


def test_parameter():
    kernel = kernels.ScaleKernel(kernels.RBFKernel())
    assert len(list(kernel.parameters())) == 2