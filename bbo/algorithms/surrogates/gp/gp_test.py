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

import pytest
import torch

from bbo.algorithms.surrogates.gp.gp import GP
from bbo.algorithms.surrogates.gp import kernels
from bbo.algorithms.surrogates.gp import kernel_impl

bs, n1, n2, d = 4, 20, 10, 3
X_32 = torch.rand((n1, d), dtype=torch.float32)
Y_32 = torch.randn((n1, 1), dtype=torch.float32)
X_64 = torch.rand((n1, d), dtype=torch.float64)
Y_64 = torch.randn((n1, 1), dtype=torch.float64)


@pytest.mark.parametrize("gp", [
    GP(X_64, Y_64, epochs=30),
    GP(X_64, Y_64, epochs=30, cov_module=kernels.Matern52Kernel()),
    GP(X_32, Y_32, epochs=30, cov_module=kernels.TemplateKernel(kernel_impl.rbf_cdist)),
    GP(X_32, Y_32, epochs=30, cov_module=kernels.TemplateKernel(kernel_impl.matern52_cdist)),
    GP(X_64, Y_64, epochs=30, cov_module=kernels.ScaleKernel(kernels.RBFKernel()))
])
class TestGPRun:
    def test_train(self, gp):
        gp.train()
    
    def test_predict_2d(self, gp):
        n_query = 10
        query_X = torch.randn((n_query, d))
        mu, var = gp.predict(query_X)
        assert mu.shape == (n_query, 1)
        assert var.shape == (n_query, n_query)

    def test_predict_3d(self, gp):
        n_query = 10
        query_X = torch.randn((bs, n_query, d))
        parallel_pred_mu, parallel_pred_var = gp.predict(query_X)
        assert parallel_pred_mu.shape == (bs, n_query, 1)
        assert parallel_pred_var.shape == (bs, n_query, n_query)

        seq_pred_mu, seq_pred_var = [], []
        for i in range(bs):
            pred_mu, pred_var = gp.predict(query_X[i])
            seq_pred_mu.append(pred_mu.unsqueeze(0))
            seq_pred_var.append(pred_var.unsqueeze(0))
        seq_pred_mu = torch.cat(seq_pred_mu, dim=0)
        seq_pred_var = torch.cat(seq_pred_var, dim=0)
        assert torch.allclose(parallel_pred_mu, seq_pred_mu)
        assert torch.allclose(parallel_pred_var, seq_pred_var)