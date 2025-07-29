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

from bbo.algorithms.surrogates.gp.means import ConstantMean


@pytest.mark.parametrize('mean', [
    ConstantMean()
])
def test_mean_2d(mean):
    X = torch.randn(10, 2)
    m = mean(X)
    assert m.shape == (10, 1)


@pytest.mark.parametrize('mean', [
    ConstantMean()
])
def test_mean_3d(mean):
    X = torch.randn(20, 10, 2)
    m = mean(X)
    assert m.shape == (20, 10, 1)