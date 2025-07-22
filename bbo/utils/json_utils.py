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
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__ndarray__': True,
                'value': obj.tolist(),
                'dtype': obj.dtype.name,
                'shape': obj.shape}
        return super().default(obj)


def numpy_hook(obj):
    if '__ndarray__' in obj:
        return np.array(obj['value'], dtype=obj['dtype']).reshape(obj['shape'])
    return obj