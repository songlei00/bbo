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

from bbo.benchmarks.experimenters.synthetic.branin import branin, Branin2DExperimenter
from bbo.shared.trial import Trial


X = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
Y = np.array([0.397887] * 3)

def test_impl():
    assert np.allclose(branin(X), Y, rtol=1e-5)

def test_experimenter():
    trials: Sequence[Trial] = []
    for i in range(len(X)):
        x1, x2 = X[i]
        trial = Trial({
            'x1': x1,
            'x2': x2,
        })
        trials.append(trial)
    exp = Branin2DExperimenter()
    exp.evaluate(trials)
    for i in range(len(trials)):
        np.allclose(trials[i].final_measurement.metrics[exp.metric_name].value, Y[i], rtol=1e-5)