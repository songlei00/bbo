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

from bbo.algorithms.surrogates.gp.objectives import neg_log_marginal_likelihood
from bbo.algorithms.surrogates.gp.means import ConstantMean
from bbo.algorithms.surrogates.gp.kernels import RBFKernel


def test_neg_log_marginal_likelihood_1():
    X = torch.randn((20, 3))
    y = torch.randn(20, 1)
    mean_module = ConstantMean()
    cov_module = RBFKernel()
    loss = neg_log_marginal_likelihood(X, y, mean_module, cov_module)
    assert loss.numel() == 1