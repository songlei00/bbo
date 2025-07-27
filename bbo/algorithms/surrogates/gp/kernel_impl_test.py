import math

import pytest
import torch
import torch.nn.functional as F

from bbo.algorithms.surrogates.gp import kernel_impl

bs, n1, n2, d = 4, 20, 10, 3


def _naive_rbf_impl(X1, X2, lengthscale=1.0):
    X1 = X1.unsqueeze(-2)
    X2 = X2.unsqueeze(-3)
    return torch.exp(-0.5 * torch.sum(((X1 - X2) / lengthscale) ** 2, dim=-1))

def _naive_matern52_impl(X1, X2, lengthscale=1.0):
    X1 = X1.unsqueeze(-2)
    X2 = X2.unsqueeze(-3)
    d = torch.sqrt(torch.sum(((X1 - X2) / lengthscale) ** 2, dim=-1))
    return (1 + math.sqrt(5) * d + 5 / 3 * d ** 2) * torch.exp(-math.sqrt(5) * d)


@pytest.mark.parametrize("naive_impl, fast_impl", [
    (_naive_rbf_impl, kernel_impl.rbf_vmap_2d),
    (_naive_matern52_impl, kernel_impl.matern52_vmap_2d),
    (_naive_rbf_impl, kernel_impl.rbf_cdist),
    (_naive_matern52_impl, kernel_impl.matern52_cdist)
])
def test_kernel_2d(naive_impl, fast_impl):
    X1 = torch.randn((n1, d))
    X2 = torch.randn((n2, d))
    lengthscale = F.softplus(torch.randn(()))
    K = fast_impl(X1, X2, lengthscale)
    ref_K = naive_impl(X1, X2, lengthscale)
    assert K.shape == (n1, n2)
    assert torch.allclose(K, ref_K)


@pytest.mark.parametrize("naive_impl, fast_impl", [
    (_naive_rbf_impl, kernel_impl.rbf_vmap_3d),
    (_naive_matern52_impl, kernel_impl.matern52_vmap_3d),
    (_naive_rbf_impl, kernel_impl.rbf_cdist),
    (_naive_matern52_impl, kernel_impl.matern52_cdist)
])
def test_kernel_3d(naive_impl, fast_impl):
    X1 = torch.randn((bs, n1, d))
    X2 = torch.randn((bs, n2, d))
    lengthscale = F.softplus(torch.randn(()))
    K = fast_impl(X1, X2, lengthscale)
    ref_K = naive_impl(X1, X2, lengthscale)
    assert K.shape == (bs, n1, n2)
    assert torch.allclose(K, ref_K)


@pytest.mark.parametrize("fast_impl", [
    kernel_impl.rbf_vmap_2d,
    kernel_impl.matern52_vmap_2d,
    kernel_impl.rbf_cdist,
    kernel_impl.matern52_cdist
])
def test_pd(fast_impl, num_run=100):
    for _ in range(num_run):
        X = torch.randn((n1, d))
        lengthscale = F.softplus(torch.randn(()))
        K = fast_impl(X, X, lengthscale) + 1e-6 * torch.eye(n1)
        try:
            torch.linalg.cholesky(K)
        except torch._C._LinAlgError:
            pytest.fail("K is not positive definite")