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

import pytest

from bbo.shared.base_study_config import (
    ObjectiveMetricGoal,
    MetricInformation,
    MetricInformationList,
    ProblemStatement
)


def test_metric_information_list():
    l = MetricInformationList()
    m1 = MetricInformation('obj1', ObjectiveMetricGoal.MAXIMIZE)
    l.append(m1)
    dup_m1 = MetricInformation('obj1', ObjectiveMetricGoal.MAXIMIZE) 
    with pytest.raises(ValueError):
        l.append(dup_m1)
    m2 = MetricInformation('obj2', ObjectiveMetricGoal.MAXIMIZE)  
    l.append(m2)
    assert len(l) == 2


def test_problem_statement():
    ps = ProblemStatement()
    metrics = ps.metric_information
    m = MetricInformation('obj', ObjectiveMetricGoal.MAXIMIZE)
    metrics.append(m)
    assert ps.is_single_objective
    assert m == ps.metric_information_item()