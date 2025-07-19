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

from collections import abc


class FrozenDict(abc.Mapping):
    def __init__(self, *args, **kwargs):
        self._data = dict(*args, **kwargs)
        self._hash = self._get_hash()

    def _get_hash(self):
        h = 0
        for k, v in self._data.items():
            h ^= hash((k, v))
        return h

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"FrozenDict({self._data})"