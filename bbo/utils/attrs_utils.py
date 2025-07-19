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

from typing import Any, Union, Tuple

import attrs


def assert_not_negative(instance: Any, attribute: attrs.Attribute, value: Union[float, int]):
    if value < 0:
        raise ValueError(f'{attribute.name} must be non-negative in {instance}')


def assert_bounds(instance: Any, attribute: attrs.Attribute, value: Union[Tuple[float, float], Tuple[int, int]]):
    if len(value) != 2:
        raise ValueError(f'{attribute.name} must be a tuple of length 2. Given {value}')
    if not isinstance(value[0], (float, int)) and not isinstance(value[1], (float, int)):
        raise ValueError(f'{attribute.name} must be a tuple of numbers. Given {value}')
    if value[0] > value[1]:
        raise ValueError(f'Low bound must be less than high bound. Given {value}')
