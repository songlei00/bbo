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

import functools
from typing import Sequence, Callable, Dict, Optional

import numpy as np


def arg_operator(data: Sequence, operator: Callable[[Sequence], int]):
    index, _ = operator(enumerate(data), key=lambda x: x[1])
    return index

argmin = functools.partial(arg_operator, operator=min)
argmax = functools.partial(arg_operator, operator=max)


def eq_dict_of_ndarray(d1: Optional[Dict[str, np.ndarray]], d2: Optional[Dict[str, np.ndarray]]):
    if d1 is None or d2 is None:
        return d1 is None and d2 is None
    if len(d1) != len(d2):
        return False
    for key in d1:
        if key not in d2 or not np.allclose(d1[key], d2[key]):
            return False
    return True