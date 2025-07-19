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

from bbo.shared.parameter_config import ParameterConfig, SearchSpace, ScaleType, ParameterType


class TestParameterConfig:
    def test_create_double(self):
        pc = ParameterConfig.factory(
            'double',
            bounds=(0.0, 1.0),
            scale_type=ScaleType.LINEAR,
            default_value=0.0
        )
        assert pc.name == 'double'
        assert pc.type == ParameterType.DOUBLE
        assert pc.bounds == (0.0, 1.0)
        with pytest.raises(ValueError):
            pc.feasible_values
        assert pc.scale_type == ScaleType.LINEAR
        assert pc.default_value == 0.0
        assert pc.is_feasible(pc.sample())

    def test_create_int(self):
        pc = ParameterConfig.factory(
            'int',
            bounds=(0, 2),
            scale_type=ScaleType.LINEAR,
            default_value=0
        )
        assert pc.name == 'int'
        assert pc.type == ParameterType.INTEGER
        assert pc.bounds == (0, 2)
        assert tuple(pc.feasible_values) == (0, 1, 2)
        assert pc.scale_type == ScaleType.LINEAR
        assert pc.default_value == 0
        assert pc.is_feasible(pc.sample())

    def test_create_discrete(self):
        pc = ParameterConfig.factory(
            'discrete',
            feasible_values=[0, 2, 4],
            scale_type=ScaleType.UNIFORM_DISCRETE,
            default_value=2
        )
        assert pc.name == 'discrete'
        assert pc.type == ParameterType.DISCRETE
        assert pc.bounds == (0, 4)
        assert tuple(pc.feasible_values) == (0, 2, 4)
        assert pc.scale_type == ScaleType.UNIFORM_DISCRETE
        assert pc.default_value == 2
        assert pc.is_feasible(pc.sample())

    def test_create_categorical(self):
        pc = ParameterConfig.factory(
            'categorical',
            feasible_values=['a', 'b', 'c'],
            default_value='b'
        )
        assert pc.name == 'categorical'
        assert pc.type == ParameterType.CATEGORICAL
        with pytest.raises(ValueError):
            pc.bounds
        assert tuple(pc.feasible_values) == ('a', 'b', 'c')
        assert pc.scale_type is None
        assert pc.default_value == 'b'
        assert pc.is_feasible(pc.sample())

    def test_wrong_default_value(self):
        with pytest.raises(ValueError):
            ParameterConfig.factory(
                'double',
                bounds=(0.0, 1.0),
                default_value=1.1
            )
        with pytest.raises(ValueError):
            ParameterConfig.factory(
                'int',
                bounds=(0, 2),
                default_value=3
            )
        with pytest.raises(ValueError):
            ParameterConfig.factory(
                'discrete',
                feasible_values=[0, 2, 4],
                default_value=3
            )
        with pytest.raises(ValueError):
            ParameterConfig.factory(
                'categorical',
                feasible_values=['a', 'b', 'c'],
                default_value='d'
            )


class TestSearchSpace:
    def test_create_double(self):
        ss = SearchSpace()
        ss.add_float_param('double', 0, 1)
        pc = ss.get('double')
        assert pc.name == 'double'
        assert pc.type == ParameterType.DOUBLE
        assert pc.bounds == (0.0, 1.0)
        with pytest.raises(ValueError):
            pc.feasible_values
        assert pc.scale_type == ScaleType.LINEAR
        assert pc.default_value is None

    def test_create_int(self):
        ss = SearchSpace()
        ss.add_int_param('int', 0, 2)
        pc = ss.get('int')
        assert pc.name == 'int'
        assert pc.type == ParameterType.INTEGER
        assert pc.bounds == (0, 2)
        assert tuple(pc.feasible_values) == (0, 1, 2)
        assert pc.scale_type is None
        assert pc.default_value is None
        
    def test_create_discrete(self):
        ss = SearchSpace()
        ss.add_discrete_param('discrete', [0, 2, 4], default_value=2)
        pc = ss.get('discrete')
        assert pc.name == 'discrete'
        assert pc.type == ParameterType.DISCRETE
        assert pc.bounds == (0, 4)
        assert tuple(pc.feasible_values) == (0, 2, 4)
        assert pc.scale_type is None
        assert pc.default_value == 2

    def test_create_categorical(self):
        ss = SearchSpace()
        ss.add_categorical_param('categorical', ['a', 'b', 'c'], default_value='b')
        pc = ss.get('categorical')
        assert pc.name == 'categorical'
        assert pc.type == ParameterType.CATEGORICAL
        with pytest.raises(ValueError):
            pc.bounds
        assert tuple(pc.feasible_values) == ('a', 'b', 'c')
        assert pc.scale_type is None
        assert pc.default_value == 'b'

    def test_create_custom(self):
        ss = SearchSpace()
        ss.add_custom_param('custom', default_value='a')
        pc = ss.get('custom')
        assert pc.name == 'custom'
        assert pc.type == ParameterType.CUSTOM

    def test_sample(self):
        ss = SearchSpace()
        ss.add_float_param('float1', 0, 1)
        ss.add_float_param('float2', 0, 1)
        ss.add_int_param('int', 0, 2)
        ss.add_discrete_param('discrete', [0, 2, 4])
        ss.add_categorical_param('categorical', ['a', 'b', 'c'])
        assert ss.num_parameters() == 5
        assert ss.num_parameters(ParameterType.DOUBLE) == 2
        assert ss.num_parameters(ParameterType.INTEGER) == 1
        assert ss.num_parameters(ParameterType.DISCRETE) == 1
        assert ss.num_parameters(ParameterType.CATEGORICAL) == 1
        assert not ss.is_conditional
        sampled = ss.sample()
        assert ss.is_feasible(sampled)
        for pc in ss.parameter_configs.values():
            assert pc.is_feasible(sampled[pc.name])

    def test_conditional_search_space(self):
        ss = SearchSpace()
        ss.add_int_param('int', 0, 2)
        ss.add_discrete_param('discrete', [0, 2, 4])
        sub_ss = ss.subspace({'int': [1], 'discrete': [2]})
        sub_ss.add_float_param('float1', 0, 3)
        sub_ss.add_float_param('float2', 0, 4)
        assert ss.is_conditional
        assert len(ss.children) == 1