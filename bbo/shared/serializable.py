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

import abc
from typing import TypeVar

from bbo.shared.metadata import Metadata

_T = TypeVar('_T')


class PartiallySerializable(abc.ABC):
    @abc.abstractmethod
    def load(self, metadata: Metadata) -> None:
        pass

    @abc.abstractmethod
    def dump(self) -> Metadata:
        pass


class Serializable(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def recover(cls: _T, metadata: Metadata) -> _T:
        pass

    @abc.abstractmethod
    def dump(self) -> Metadata:
        pass