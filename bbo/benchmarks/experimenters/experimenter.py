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
from typing import Sequence

from bbo.shared.trial import Trial
from bbo.shared.base_study_config import ProblemStatement


class Experimenter(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, trials: Sequence[Trial]):
        pass

    @abc.abstractmethod
    def problem_statement(self) -> ProblemStatement:
        pass