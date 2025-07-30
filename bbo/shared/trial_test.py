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

from bbo.shared.base_study_config import ProblemStatement, MetricInformation, ObjectiveMetricGoal
from bbo.shared.trial import (
    Metric,
    MetricDict,
    ParameterValue,
    ParameterDict,
    Trial,
    Measurement,
    TrialStatus,
    trial_is_better_than
)


def test_metric_dict():
    metric_dict = MetricDict({'obj1': 1})
    metric_dict['obj2'] = 2
    metric_dict['obj3'] = Metric(3, 1)
    assert len(metric_dict) == 3
    assert metric_dict.get_value('obj1') == 1
    assert metric_dict.get_float_dict() == {'obj1': 1, 'obj2': 2, 'obj3': 3}


def test_parameter_dict():
    parameter_dict = ParameterDict({'f1': 1})
    parameter_dict['f2'] = 2
    parameter_dict['f3'] = ParameterValue(3)
    assert len(parameter_dict) == 3
    assert parameter_dict.get_value('f1') == 1
    assert parameter_dict.get_float_dict() == {'f1': 1, 'f2': 2, 'f3': 3}


class TestTrial:
    def test_trial(self):
        trial = Trial({'f1': 1, 'f2': 2})
        assert trial.status == TrialStatus.ACTIVE
        measurement = Measurement({'obj': 1})
        trial.complete(measurement)
        assert trial.status == TrialStatus.COMPLETED
        assert trial.duration.total_seconds() > 0
        assert not trial.infeasible

    def test_infeasible(self):
        trial = Trial({'f1': 1, 'f2': 2})
        trial.complete(None, infeasibility_reason='test')
        assert trial.status == TrialStatus.COMPLETED
        assert trial.duration.total_seconds() > 0
        assert trial.infeasible


@pytest.mark.parametrize('trial1, trial2, ps, expected', [
    (
        Trial({}, final_measurement=Measurement({'obj': 0.0})),
        Trial({}, final_measurement=Measurement({'obj': 1.0})),
        ProblemStatement(
            metric_information=[MetricInformation(name='obj', goal=ObjectiveMetricGoal.MAXIMIZE)]
        ),
        False
    ),
    (
        Trial({}, final_measurement=Measurement({'obj': 1.0})),
        Trial({}, final_measurement=Measurement({'obj': 0.0})),
        ProblemStatement(
            metric_information=[MetricInformation(name='obj', goal=ObjectiveMetricGoal.MAXIMIZE)]
        ),
        True
    ),
    (
        Trial({}, final_measurement=Measurement({'obj': 0.0})),
        Trial({}, final_measurement=Measurement({'obj': 0.0})),
        ProblemStatement(
            metric_information=[MetricInformation(name='obj', goal=ObjectiveMetricGoal.MAXIMIZE)]
        ),
        False
    ),
    (
        Trial({}, final_measurement=Measurement({'obj': 0.0})),
        Trial({}, final_measurement=Measurement({'obj': 1.0})),
        ProblemStatement(metric_information=[
            MetricInformation(name='obj', goal=ObjectiveMetricGoal.MINIMIZE)
        ]),
        True
    ),
    (
        Trial({}, final_measurement=Measurement({'obj': 1.0})),
        Trial({}, final_measurement=Measurement({'obj': 0.0})),
        ProblemStatement(metric_information=[
            MetricInformation(name='obj', goal=ObjectiveMetricGoal.MINIMIZE)
        ]),
        False
    ),
    (
        Trial({}, final_measurement=Measurement({'obj': 0.0})),
        Trial({}, final_measurement=Measurement({'obj': 0.0})),
        ProblemStatement(metric_information=[
            MetricInformation(name='obj', goal=ObjectiveMetricGoal.MINIMIZE)
        ]),
        False
    ),
])
def test_trial_is_better_than(
    trial1: Trial,
    trial2: Trial,
    ps: ProblemStatement,
    expected: bool
):
    assert trial_is_better_than(trial1, trial2, ps) == expected