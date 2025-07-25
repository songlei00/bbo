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

from bbo.utils.frozendict_utils import FrozenDict


def test_frozendict():
    d = FrozenDict({'a': 1, 'b': 2})
    assert d['a'] == 1 and d['b'] == 2
    with pytest.raises(TypeError):
        d['a'] = 3
    with pytest.raises(TypeError):
        d['c'] = 3
    with pytest.raises(KeyError):
        d['c']
    e = FrozenDict({'a': 1, 'b': 2}) 
    assert d == e