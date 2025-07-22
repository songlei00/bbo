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
import copy

import numpy as np
from attrs import define, field, validators

from bbo.algorithms.designers.evolution import templates
from bbo.algorithms.converters.core import NumpyArraySpecType
from bbo.utils import attrs_utils


@define
class RandomMutation(templates.Mutation):
    _converter: templates.PopulationConverter = field(
        validator=validators.instance_of(templates.PopulationConverter)
    )
    _k: int = field(
       default=1,
       validator=validators.and_(validators.instance_of(int), attrs_utils.assert_positive)
    )
    
    def __call__(self, population: templates.Population) -> templates.Population:
        assert len(population) == 1
        ret = copy.deepcopy(population)
        ret.ages[0] = 0
        output_specs = self._converter.trial_converter.output_specs
        keys = random.sample(list(population.xs), k=self._k)
        for k in keys:
            output_spec = output_specs[k]
            lb, ub = output_spec.bounds
            shape = (1, 1)
            if output_spec.type == NumpyArraySpecType.DOUBLE:
                ret.xs[k] = np.random.rand(*shape) * (ub - lb) + lb
            elif output_spec.type in (
                NumpyArraySpecType.INTEGER,
                NumpyArraySpecType.DISCRETE,
                NumpyArraySpecType.CATEGORICAL
            ):
                ret.xs[k] = np.random.randint(lb, ub+1, shape)
        return ret