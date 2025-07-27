import functools
import math
from typing import Protocol, Union

import torch


class KernelProtocol(Protocol):
    def __call__(self, X1: torch.Tensor, X2: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass


# === Kernel implementations ===

def _rbf_impl(r2: torch.Tensor):
    ret = torch.exp(-0.5 * r2)
    return ret

def _matern52_impl(r: torch.Tensor):
    return (1 + r + r**2 / 3) * torch.exp(-r)


# === torch.cdist implementations ===

def rbf_cdist(
    X1: torch.Tensor,
    X2: torch.Tensor,
    lengthscale: Union[float, torch.Tensor]
) -> torch.Tensor:
    r2 = torch.cdist(X1 / lengthscale, X2 / lengthscale).square()
    return _rbf_impl(r2)

def matern52_cdist(
    X1: torch.Tensor,
    X2: torch.Tensor,
    lengthscale: Union[float, torch.Tensor]
):
    r = math.sqrt(5) * torch.cdist(X1 / lengthscale, X2 / lengthscale)
    return _matern52_impl(r)


# === torch.vmap implementations ===

pairwise_dis = torch.nn.PairwiseDistance()

def cov_matrix_vmap_2d(cov_fn: KernelProtocol):
    @functools.wraps(cov_fn)
    def cov_matrix(X1, X2, *args, **kwargs):
        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("X1 and X2 must be 2d tensors. Got {} and {}.".format(X1.shape, X2.shape))
        mmap = torch.vmap(lambda x: torch.vmap(lambda y: cov_fn(x, y, *args, **kwargs))(X1))
        return mmap(X2).T
    return cov_matrix

def cov_matrix_vmap_3d(cov_fn: KernelProtocol):
    cov_matrix = cov_matrix_vmap_2d(cov_fn)
    @functools.wraps(cov_matrix)
    def batch_cov_matrix(X1, X2, *args, **kwargs):
        if X1.ndim != 3 or X2.ndim != 3:
            raise ValueError("X1 and X2 must be 3d tensors. Got {} and {}.".format(X1.shape, X2.shape))
        return torch.vmap(lambda X1, X2: cov_matrix(X1, X2, *args, **kwargs))(X1, X2)
    return batch_cov_matrix

def _rbf(
    X1: torch.Tensor,
    X2: torch.Tensor,
    lengthscale: Union[float, torch.Tensor]
) -> torch.Tensor:
    r2 = torch.sum(((X1 - X2) / (lengthscale)) ** 2)
    return _rbf_impl(r2)

def _matern52(
    X1: torch.Tensor,
    X2: torch.Tensor,
    lengthscale: Union[float, torch.Tensor]
) -> torch.Tensor:
    scaled_diff = (X1 - X2) / lengthscale
    dist = pairwise_dis(scaled_diff, torch.zeros_like(scaled_diff))
    r = math.sqrt(5) * dist
    return _matern52_impl(r)

rbf_vmap_2d = cov_matrix_vmap_2d(_rbf)
rbf_vmap_3d = cov_matrix_vmap_3d(_rbf)
matern52_vmap_2d = cov_matrix_vmap_2d(_matern52)
matern52_vmap_3d = cov_matrix_vmap_3d(_matern52)