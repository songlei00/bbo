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
from attrs import define, field
import numpy as np

from bbo.utils import attrs_utils


def assertValidatorWorksAsIntended(validator, value, result: bool):
    @define
    class Test:
        x = field(validator=validator)
    
    if result:
        Test(value)
    else:
        with pytest.raises(ValueError):
            Test(value)


@pytest.mark.parametrize(
    'validator, value, result', [
        (attrs_utils.assert_not_negative, -1, False),
        (attrs_utils.assert_not_negative, -0.1, False),
        (attrs_utils.assert_not_negative, 0, True),
        (attrs_utils.assert_not_negative, 1, True),
        (attrs_utils.assert_not_negative, 0.1, True)
    ]
)
def test_assert_not_negative(validator, value, result: bool):
    assertValidatorWorksAsIntended(validator, value, result)


@pytest.mark.parametrize(
    'validator, value, result', [
        (attrs_utils.assert_bounds, (0, 1), True),
        (attrs_utils.assert_bounds, tuple(), False),
        (attrs_utils.assert_bounds, (0, ), False),
        (attrs_utils.assert_bounds, (1, 2, 3), False),
        (attrs_utils.assert_bounds, (1, 0), False),
        (attrs_utils.assert_bounds, ('0', '1'), False)
    ]
)
def test_assert_bounds(validator, value, result: bool):
    assertValidatorWorksAsIntended(validator, value, result)


@pytest.mark.parametrize(
    'validator, value, result', [
        (attrs_utils.assert_positive, -1, False),
        (attrs_utils.assert_positive, 0, False),
        (attrs_utils.assert_positive, 1, True),
        (attrs_utils.assert_positive, 0.1, True)
    ]
)
def test_assert_positive(validator, value, result: bool):
    assertValidatorWorksAsIntended(validator, value, result)


@pytest.mark.parametrize(
    'validator, value, result', [
        (attrs_utils.shape_equals(lambda x: (len(x), None)), np.zeros((3, 5)), True),
        (attrs_utils.shape_equals(lambda x: (len(x), )), np.zeros((3, 5)), False),
        (attrs_utils.shape_equals(lambda x: (len(x), )), np.zeros((3, )), True),
    ]
)
def test_shape_equals(validator, value, result: bool):
    assertValidatorWorksAsIntended(validator, value, result)