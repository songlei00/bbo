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

from typing import Union

import torch
import torch.nn as nn
from attrs import define, field, validators

from bbo.algorithms.abstractions import Surrogate
from bbo.utils import attrs_utils, logging_utils
from bbo.algorithms.surrogates.gp.means import ConstantMean
from bbo.algorithms.surrogates.gp.kernels import Matern52Kernel
from bbo.algorithms.surrogates.gp.objectives import neg_log_marginal_likelihood, solve_gp_linear_system

logger = logging_utils.get_logger(__name__)


@define
class GP(Surrogate):
    _X: torch.Tensor = field(validator=attrs_utils.assert_2dtensor)
    _Y: torch.Tensor = field(validator=attrs_utils.assert_2dtensor)
    _mean_module: nn.Module = field(factory=lambda: ConstantMean())
    _cov_module: nn.Module = field(factory=lambda: Matern52Kernel())
    _device: str = field(default='cpu', kw_only=True)

    # Training config
    _lr: float = field(default=0.01, converter=float, validator=validators.instance_of(float), kw_only=True)
    _epochs: int = field(default=100, converter=int, validator=validators.instance_of(int), kw_only=True)
    _noise_variance: float = field(default=1e-6, converter=float, validator=validators.instance_of(float), kw_only=True)

    _objective = field(default=neg_log_marginal_likelihood, init=False)
    _cached_chol: torch.Tensor | None = field(default=None, init=False)
    _cached_kinvy: torch.Tensor | None = field(default=None, init=False)

    def __attrs_post_init__(self):
        if isinstance(self._device, str) and self._device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"Device {self._device} is not available. Use CPU")
        self._device = torch.device(self._device if torch.cuda.is_available() else 'cpu')
        self._mean_module = self._mean_module.to(self._device)
        self._cov_module = self._cov_module.to(self._device)

    def train(self):
        params = list(self._mean_module.parameters()) + list(self._cov_module.parameters())
        optimizer = torch.optim.Adam(params, lr=self._lr)
        for epoch in range(self._epochs):
            loss = self._objective(
                self._X, self._Y, self._mean_module, self._cov_module, self._noise_variance
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, query_X):
        if self._cached_chol is None or self._cached_kinvy is None:
            with torch.no_grad():
                self._cached_chol, self._cached_kinvy, _ = solve_gp_linear_system(
                    self._X,
                    self._Y,
                    self._mean_module,
                    self._cov_module,
                    self._noise_variance
                )
        chol, kinv = self._cached_chol, self._cached_kinvy
        cov = self._cov_module(query_X, self._X)
        mu = self._mean_module(query_X) + cov @ kinv
        if cov.ndim == 3:
            chol = chol.unsqueeze(0)
        v = torch.linalg.solve_triangular(chol, cov.transpose(-2, -1), upper=False)
        var = self._cov_module(query_X, query_X) - v.transpose(-2, -1) @ v + \
            self._noise_variance * torch.eye(query_X.shape[0]).to(query_X.device)
        return mu, var
    