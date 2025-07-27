import math
import torch
import torch.nn as nn


def solve_gp_linear_system(
    X: torch.Tensor,
    Y: torch.Tensor,
    mean_module: nn.Module,
    cov_module: nn.Module,
    warp_module=None,
    noise_variance: float = 1e-6
):
    delta_Y = Y - mean_module(X)
    K = cov_module(X, X) + torch.eye(X.shape[0]) * (noise_variance + 1e-6)
    chol = torch.linalg.cholesky(K, upper=False)
    kinvy = torch.cholesky_solve(delta_Y, chol, upper=False)
    return chol, kinvy, delta_Y


def neg_log_marginal_likelihood(
    X: torch.Tensor,
    y: torch.Tensor,
    mean_module: nn.Module,
    cov_module: nn.Module,
    warp_module=None,
    noise_variance: float = 1e-6
) -> torch.Tensor:
    chol, kinvy, delta_Y = solve_gp_linear_system(X, y, mean_module, cov_module, warp_module, noise_variance)
    nll = - 0.5 * delta_Y.T @ kinvy \
        - torch.sum(torch.log(torch.diag(chol))) \
        - 0.5 * len(y) * math.log(2 * math.pi)
    return nll