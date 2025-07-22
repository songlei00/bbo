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
import numpy as np

from bbo.utils import math_utils


@pytest.mark.parametrize(
    'func, data, expected',
    [
        (math_utils.argmin, [1, 2, 3], 0),
        (math_utils.argmax, [1, 2, 3], 2),
    ]
)
def test_operator(func, data, expected):
    assert func(data) == expected


@pytest.mark.parametrize(
    'd1, d2, expected',
    [
        ({'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, True),
        ({'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 7])}, False),
        ({'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, {'a': np.array([1, 2, 3])}, False),
        ({'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, {'a': np.array([1, 2, 3]), 'c': np.array([4, 5, 6])}, False),
    ]
)
def test_eq_dict_of_ndarray(d1, d2, expected):
    assert math_utils.eq_dict_of_ndarray(d1, d2) == expected