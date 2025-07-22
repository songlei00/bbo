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

from typing import Optional, Sequence

from attrs import define, field, validators
import numpy as np

from bbo.algorithms.abstractions import Designer, CompletedTrials, ActiveTrials
from bbo.shared.base_study_config import ProblemStatement
from bbo.shared.trial import Trial


@define
class RandomDesigner(Designer):
    problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    dtype: np.dtype = field(default=np.float32)

    def suggest(self, count: Optional[int] = None) -> Sequence[Trial]:
        count = count or 1
        suggestions = []
        for _ in range(count):
            sample = self.problem_statement.search_space.sample()
            suggestions.append(Trial(parameters=sample))
        return suggestions

    def update(self, completed_trials: CompletedTrials, active_trials: Optional[ActiveTrials] = None):
        pass