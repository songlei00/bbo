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

import numpy as np

from bbo.algorithms.converters.core import (
    NumpyArraySpecType,
    DictOf2DArray,
    DefaultModelOutputConverter,
    DefaultTrialConverter,
    TrialToArrayConverter,
    TrialToTypeArrayConverter
)
from bbo.shared.trial import Measurement
from bbo.shared.base_study_config import ProblemStatement, MetricInformation, ObjectiveMetricGoal
from bbo.utils.testing import create_dummy_ps, generate_trials, compare_trials_xy

trial_size = 5
ps, trials, cardinality = create_dummy_ps(trial_size)
active_trials = generate_trials(ps, trial_size, False)


def test_dict_of_2d_array():
    d = {'a': [[1, 2], [3, 4]], 'b': [[5,], [6,]]}
    d_2darray = DictOf2DArray(d)
    with pytest.raises(ValueError):
        d_2darray['c'] = [[1], [2], [3]]
    array = d_2darray.to_array()
    assert array.tolist() == [[1, 2, 5], [3, 4, 6]]
    assert d_2darray.to_dict(array) == d_2darray


class TestFlipSign:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.name = 'obj'
        self.n = 10
        self.measurements = [Measurement({self.name: i}) for i in range(self.n)]

    def test_flip_sign_for_minimization_metrics(self):
        metric_info = MetricInformation(self.name, ObjectiveMetricGoal.MINIMIZE)
        output_converter = DefaultModelOutputConverter(metric_info, flip_sign_for_minimization_metrics=True)

        # Test converter
        labels = output_converter.convert(self.measurements)
        assert (labels == -np.asarray([i for i in range(self.n)]).reshape(-1, 1)).all()
        
        metrics = output_converter.to_metrics(labels)
        for i in range(self.n):
            assert metrics[i].value == i

        # Test metric information
        fliped_metric_info = output_converter.metric_information
        assert fliped_metric_info.goal.is_maximize
        assert metric_info.goal.is_minimize

    def test_no_flip_sign(self):
        metric_info = MetricInformation(self.name, ObjectiveMetricGoal.MINIMIZE)
        output_converter = DefaultModelOutputConverter(metric_info)
        labels = output_converter.convert(self.measurements)
        assert (labels == np.asarray([i for i in range(self.n)]).reshape(-1, 1)).all()
        assert output_converter.metric_information == metric_info


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
        compare_trials_xy(ts, trials)

    def test_to_trials_without_y(self):
        converter = DefaultTrialConverter.from_study_config(ps)
        x = converter.to_features(active_trials)
        ts = converter.to_trials(x)
        compare_trials_xy(ts, active_trials)

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
        compare_trials_xy(ts, trials)

    def test_to_trials_without_y(self):
        converter = TrialToArrayConverter.from_study_config(ps)
        x = converter.to_features(active_trials)
        ts = converter.to_trials(x)
        compare_trials_xy(ts, active_trials)

    def test_onehot(self):
        converter = TrialToArrayConverter.from_study_config(ps, onehot_embed=True)
        x, y = converter.to_xy(trials)
        assert x.shape == (trial_size, 1 + 1 + cardinality['discrete'] + cardinality['categorical'])
        assert y.shape == (trial_size, 1)


class TestTrialToTypeArrayConverter:
    def test_default(self):
        converter = TrialToTypeArrayConverter.from_study_config(ps)
        x, y = converter.to_xy(trials)
        assert x.double.shape == (trial_size, 1)
        assert x.integer.shape == (trial_size, 1)
        assert x.discrete.shape == (trial_size, 1)
        assert x.categorical.shape == (trial_size, 1)
        assert y.shape == (trial_size, 1)
        ts = converter.to_trials(x, y)
        compare_trials_xy(ts, trials)

    def test_to_trials_without_y(self):
        converter = TrialToTypeArrayConverter.from_study_config(ps)
        x = converter.to_features(active_trials)
        ts = converter.to_trials(x)
        compare_trials_xy(ts, active_trials)

    def test_double_only(self):
        ps = ProblemStatement()
        ps.search_space.add_float_param('float1', 0, 5)
        ps.search_space.add_float_param('float2', 0, 3)
        ps.metric_information.append(MetricInformation('obj', ObjectiveMetricGoal.MAXIMIZE))
        trials = generate_trials(ps, trial_size)
        converter = TrialToTypeArrayConverter.from_study_config(ps)
        x, y = converter.to_xy(trials)
        assert x.double.shape == (trial_size, 2)
        assert x.integer.shape == (trial_size, 0)
        assert x.discrete.shape == (trial_size, 0)
        assert x.categorical.shape == (trial_size, 0)
        assert y.shape == (trial_size, 1)
        ts = converter.to_trials(x, y)
        compare_trials_xy(ts, trials)