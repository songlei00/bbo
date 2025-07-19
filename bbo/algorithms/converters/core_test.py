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
import random
import math
from typing import Sequence

import numpy as np

from bbo.algorithms.converters.core import (
    NumpyArraySpecType,
    DictOf2DArray,
    DefaultTrialConverter,
    TrialToArrayConverter
)
from bbo.shared.trial import Measurement, Trial
from bbo.shared.base_study_config import ObjectiveMetricGoal, MetricInformation, ProblemStatement


def test_dict_of_2d_array():
    d = {'a': [[1, 2], [3, 4]], 'b': [[5,], [6,]]}
    d_2darray = DictOf2DArray(d)
    with pytest.raises(ValueError):
        d_2darray['c'] = [[1], [2], [3]]
    array = d_2darray.to_array()
    assert array.tolist() == [[1, 2, 5], [3, 4, 6]]
    assert d_2darray.to_dict(array) == d_2darray


ps = ProblemStatement()
ps.search_space.add_float_param('float', 0, 5)
ps.search_space.add_int_param('int', 3, 7)
ps.search_space.add_discrete_param('discrete', [1, 3, 5])
ps.search_space.add_categorical_param('categorical', ['a', 'b'])
cardinality = dict()
for type in ['int', 'discrete', 'categorical']:
    cardinality[type] = len(ps.search_space.get(type).feasible_values)
ps.metric_information.append(MetricInformation('obj', ObjectiveMetricGoal.MAXIMIZE))
trial_size = 5
trials = []
for _ in range(trial_size):
    p = ps.search_space.sample()
    t = Trial(p)
    t.complete(Measurement({'obj': random.random()}))
    trials.append(t)


def _compare_trials(trials1: Sequence[Trial], trials2: Sequence[Trial]):
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
        
        m_d1 = t1.final_measurement.metrics.get_float_dict()
        m_d2 = t2.final_measurement.metrics.get_float_dict()
        assert m_d1.keys() == m_d2.keys()
        for k in m_d1.keys():
            assert math.isclose(m_d1[k], m_d2[k], rel_tol=1e-5)

class TestDefaultTrialConverter:
    def test_default(self):
        converter = DefaultTrialConverter.from_study_config(ps)
        x, y = converter.to_xy(trials)

        # check output_spec
        float_spec = converter.output_specs['float']
        assert float_spec.type == NumpyArraySpecType.DOUBLE
        assert float_spec.bounds == (0, 1)
        assert float_spec.num_dimensions == 1
        assert float_spec.dtype == np.float32
        for type, output_type in zip(
            ['int', 'discrete', 'categorical'],
            [NumpyArraySpecType.INTEGER, NumpyArraySpecType.DISCRETE, NumpyArraySpecType.CATEGORICAL]
        ):
            spec = converter.output_specs[type]
            assert spec.type == output_type
            assert spec.bounds == (0, cardinality[type]-1)
            assert spec.num_dimensions == 1

        # check value
        assert x['float'].shape == (trial_size, 1)
        assert np.all(x['float'] >= 0)
        assert np.all(x['float'] <= 1)
        for type in ['int', 'discrete', 'categorical']:
            assert x[type].shape == (trial_size, 1)
            assert np.all(x[type] >= 0)
            assert np.all(x[type] < cardinality[type])
        assert y['obj'].shape == (trial_size, 1)

        # check to_trials
        ts = converter.to_trials(x, y)
        _compare_trials(ts, trials)

    def test_onehot(self):
        converter = DefaultTrialConverter.from_study_config(ps, onehot_embed=True)
        x, y = converter.to_xy(trials)
        output_specs = converter.output_specs

        # check output_spec
        assert output_specs['discrete'].num_dimensions == cardinality['discrete']
        assert output_specs['categorical'].num_dimensions == cardinality['categorical']
        for type in ['float', 'int']:
            assert x[type].shape == (trial_size, 1)
        for type in ['discrete', 'categorical']:
            assert x[type].shape == (trial_size, cardinality[type])

    def test_onehot_oovs(self):
        converter = DefaultTrialConverter.from_study_config(ps, onehot_embed=True, pad_oovs=True)
        x, y = converter.to_xy(trials)
        output_specs = converter.output_specs

        # check output_spec
        assert output_specs['discrete'].num_dimensions == cardinality['discrete'] + 1
        assert output_specs['categorical'].num_dimensions == cardinality['categorical'] + 1
        for type in ['float', 'int']:
            assert x[type].shape == (trial_size, 1)
        for type in ['discrete', 'categorical']:
            assert x[type].shape == (trial_size, cardinality[type]+1)

    def test_continuify(self):
        converter = DefaultTrialConverter.from_study_config(ps, max_discrete_indices=3)
        x, y = converter.to_xy(trials)
        output_specs = converter.output_specs

        int_spec = output_specs['int']
        assert int_spec.type == NumpyArraySpecType.DOUBLE
        assert int_spec.bounds == (0, 1)
        
        discrete_spec = output_specs['discrete']
        assert discrete_spec.type == NumpyArraySpecType.DISCRETE
        assert discrete_spec.bounds == (0, cardinality['discrete']-1)

        assert x['int'].shape == (trial_size, 1)
        assert np.all(x['int'] >= 0)
        assert np.all(x['int'] <= 1)


class TestTrialToArrayConverter:
    def test_default(self):
        converter = TrialToArrayConverter.from_study_config(ps)
        x, y = converter.to_xy(trials)
        assert x.shape == (trial_size, 4)
        assert y.shape == (trial_size, 1)

        ts = converter.to_trials(x, y)
        _compare_trials(ts, trials)

    def test_onehot(self):
        converter = TrialToArrayConverter.from_study_config(ps, onehot_embed=True)
        x, y = converter.to_xy(trials)
        assert x.shape == (trial_size, 1 + 1 + cardinality['discrete'] + cardinality['categorical'])
        assert y.shape == (trial_size, 1)