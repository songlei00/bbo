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

import json
import pytest
import numpy as np

from bbo.utils.json_utils import NumpyEncoder, numpy_hook


@pytest.mark.parametrize('shape', [(3, 0), (3, 5)])
def test_numpy_serialize(shape):
    data = {
        'a': [1, 2, 3],
        'b': np.random.rand(*shape)
    }
    dumped = json.dumps(data, cls=NumpyEncoder)
    loaded = json.loads(dumped, object_hook=numpy_hook)
    assert data['a'] == loaded['a']
    assert np.allclose(data['b'], loaded['b'], 1e-5)
    assert data['b'].dtype == loaded['b'].dtype
    assert data['b'].shape == loaded['b'].shape