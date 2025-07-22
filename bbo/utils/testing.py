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

import random
import math
from typing import Sequence

from bbo.shared.base_study_config import ProblemStatement, MetricInformation, ObjectiveMetricGoal
from bbo.shared.trial import Trial, Measurement


def generate_trials(ps: ProblemStatement, size: int, completed: bool = True):
    trial_size = size
    trials = []
    for _ in range(trial_size):
        p = ps.search_space.sample()
        t = Trial(p)
        if completed:
            t.complete(Measurement({'obj': random.random()}))
        trials.append(t)
    return trials


def create_dummy_ps(size: int):
    ps = ProblemStatement()
    ps.search_space.add_float_param('float', 0, 5)
    ps.search_space.add_int_param('int', 3, 7)
    ps.search_space.add_discrete_param('discrete', [1, 3, 5])
    ps.search_space.add_categorical_param('categorical', ['a', 'b'])
    cardinality = dict()
    for type in ['int', 'discrete', 'categorical']:
        cardinality[type] = len(ps.search_space.get(type).feasible_values)
    ps.metric_information.append(MetricInformation('obj', ObjectiveMetricGoal.MAXIMIZE))
    trials = generate_trials(ps, size)
    return ps, trials, cardinality


def compare_trials(trials1: Sequence[Trial], trials2: Sequence[Trial]):
    assert len(trials1) == len(trials2)
    for t1, t2 in zip(trials1, trials2):
        param_d1 = t1.parameters.get_float_dict()
        param_d2 = t2.parameters.get_float_dict()
        assert param_d1.keys() == param_d2.keys()
        for k in param_d1.keys():
            if isinstance(param_d1[k], float):
                assert math.isclose(param_d1[k], param_d2[k], rel_tol=1e-5)
            else:
                assert param_d1[k] == param_d2[k]
        
        assert t1.is_completed == t2.is_completed
        if t1.is_completed:
            m_d1 = t1.final_measurement.metrics.get_float_dict()
            m_d2 = t2.final_measurement.metrics.get_float_dict()
            assert m_d1.keys() == m_d2.keys()
            for k in m_d1.keys():
                assert math.isclose(m_d1[k], m_d2[k], rel_tol=1e-5)