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

from bbo.algorithms.surrogates.gp.gp import GP

bs, n1, n2, d = 4, 20, 10, 3
X = torch.rand((n1, d))
Y = torch.randn((n1, 1))
gp = GP(X, Y, epochs=30)


class TestGP:
    def test_train(self):
        gp.train()

    def test_predict_2d(self):
        n_query = 10
        query_X = torch.randn((n_query, d))
        mu, var = gp.predict(query_X)
        assert mu.shape == (n_query, 1)
        assert var.shape == (n_query, n_query)

    def test_predict_3d(self):
        n_query = 10
        query_X = torch.randn((bs, n_query, d))
        vmap_predict = torch.vmap(gp.predict)
        mu, var = vmap_predict(query_X)
        assert mu.shape == (bs, n_query, 1)
        assert var.shape == (bs, n_query, n_query)