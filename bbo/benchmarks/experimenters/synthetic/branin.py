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

from typing import Sequence

import numpy as np

from bbo.benchmarks.experimenters.experimenter import Experimenter
from bbo.benchmarks.experimenters.numpy_experimenter import NumpyExperimenter
from bbo.shared.base_study_config import ProblemStatement, MetricInformation, ObjectiveMetricGoal
from bbo.shared.trial import Trial


def branin(x: np.ndarray) -> np.ndarray:
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    x1 = x[..., 0]
    x2 = x[..., 1]

    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return y


class Branin2DExperimenter(Experimenter):
    """https://www.sfu.ca/~ssurjano/branin.html"""

    def __init__(self):
        self.metric_name = 'obj'
        self._impl = NumpyExperimenter(branin, self.problem_statement())

    def evaluate(self, trials: Sequence[Trial]):
        return self._impl.evaluate(trials)

    def problem_statement(self) -> ProblemStatement:
        ps = ProblemStatement()
        ps.search_space.add_float_param('x1', -5, 10)
        ps.search_space.add_float_param('x2', 0, 15)
        ps.metric_information.append(MetricInformation(self.metric_name, ObjectiveMetricGoal.MINIMIZE))
        return ps