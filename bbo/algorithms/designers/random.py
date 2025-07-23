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
from bbo.algorithms.converters.core import DefaultTrialConverter, NumpyArraySpecType
from bbo.shared.base_study_config import ProblemStatement
from bbo.shared.trial import Trial
from bbo.utils import get_rng


@define
class RandomDesigner(Designer):
    problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    seed_or_rng: np.random.Generator | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((np.random.Generator, int)))
    )

    def __attrs_post_init__(self):
        self._rng = get_rng(self.seed_or_rng)
        self._converter = DefaultTrialConverter.from_study_config(self.problem_statement)

    def suggest(self, count: Optional[int] = None) -> Sequence[Trial]:
        count = count or 1
        sample = dict()
        for name, output_spec in self._converter.output_specs.items():
            lb, ub = output_spec.bounds
            if output_spec.type == NumpyArraySpecType.DOUBLE:
                sample[name] = self._rng.random((count, 1), output_spec.dtype) * (ub - lb) + lb
            elif output_spec.type in (
                NumpyArraySpecType.INTEGER,
                NumpyArraySpecType.CATEGORICAL,
                NumpyArraySpecType.DISCRETE
            ):
                sample[name] = self._rng.integers(lb, ub+1, (count, 1), output_spec.dtype)
            else:
                raise ValueError(f"Unsupported type {output_spec.type}")
        suggestions = [Trial(p) for p in self._converter.to_parameters(sample)]
        return suggestions

    def update(self, completed_trials: CompletedTrials, active_trials: Optional[ActiveTrials] = None):
        pass