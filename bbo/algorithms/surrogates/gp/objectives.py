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

import math
import torch
import torch.nn as nn


def solve_gp_linear_system(
    X: torch.Tensor,
    Y: torch.Tensor,
    mean_module: nn.Module,
    cov_module: nn.Module,
    noise_variance: float = 1e-6
):
    delta_Y = Y - mean_module(X)
    K = cov_module(X, X) + torch.eye(X.shape[0]) * noise_variance
    chol = torch.linalg.cholesky(K, upper=False)
    kinvy = torch.cholesky_solve(delta_Y, chol, upper=False)
    return chol, kinvy, delta_Y


def neg_log_marginal_likelihood(
    X: torch.Tensor,
    y: torch.Tensor,
    mean_module: nn.Module,
    cov_module: nn.Module,
    noise_variance: float = 1e-6
) -> torch.Tensor:
    chol, kinvy, delta_Y = solve_gp_linear_system(X, y, mean_module, cov_module, noise_variance)
    nll = - 0.5 * delta_Y.T @ kinvy \
        - torch.sum(torch.log(torch.diag(chol))) \
        - 0.5 * len(y) * math.log(2 * math.pi)
    return nll