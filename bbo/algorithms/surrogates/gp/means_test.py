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

from bbo.algorithms.surrogates.gp.warpers import MLPWarp
from bbo.algorithms.surrogates.gp.means import ConstantMean, MLPMean

bs, n, d = 20, 10, 5


@pytest.mark.parametrize('mean', [
    ConstantMean(),
    MLPMean(16, MLPWarp(d, [16, 16]))
])
class TestMeanRun:
    def test_mean_2d(self, mean):
        X = torch.randn(n, d)
        m = mean(X)
        assert m.shape == (n, 1)

    def test_mean_3d(self, mean):
        X = torch.randn(bs, n, d)
        m = mean(X)
        assert m.shape == (bs, n, 1)
