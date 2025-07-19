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

from typing import Callable, Sequence

import numpy as np
from attrs import define, field, validators

from bbo.algorithms.converters.core import TrialConverter, TrialToArrayConverter
from bbo.benchmarks.experimenters.experimenter import Experimenter
from bbo.shared.base_study_config import ProblemStatement
from bbo.shared.trial import Trial, Measurement


@define
class NumpyExperimenter(Experimenter):
    _impl: Callable[[np.ndarray], np.ndarray] = field()
    _problem_statement: ProblemStatement = field(validator=validators.instance_of(ProblemStatement))

    _metric_name: str = field(init=False)
    _converter: TrialConverter = field(init=False)

    def __attrs_post_init__(self):
        self._metric_name = self._problem_statement.metric_information_item().name
        self._converter = TrialToArrayConverter.from_study_config(
            self._problem_statement,
            scale=False
        )

    def evaluate(self, trials: Sequence[Trial]):
        features = self._converter.to_features(trials)
        results = self._impl(features)
        for t, result in zip(trials, results.flatten()):
            t.complete(Measurement({self._metric_name: result}))
    
    def problem_statement(self):
        return self._problem_statement