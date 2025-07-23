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
    _rng: np.random.Generator = field(validator=validators.instance_of(np.random.Generator))
    _k: int = field(
       default=1,
       validator=validators.and_(validators.instance_of(int), attrs_utils.assert_positive)
    )
    
    def __call__(self, population: templates.Population) -> templates.Population:
        assert len(population) == 1
        ret = templates.Population(population.xs.copy(), generations=population.generations.copy())
        output_specs = self._converter.trial_converter.output_specs
        keys = self._rng.choice(list(population.xs), size=self._k, replace=False)
        for k in keys:
            output_spec = output_specs[k]
            lb, ub = output_spec.bounds
            shape = (1, 1)
            if output_spec.type == NumpyArraySpecType.DOUBLE:
                ret.xs[k] = self._rng.random(shape, output_spec.dtype) * (ub - lb) + lb
            elif output_spec.type in (
                NumpyArraySpecType.INTEGER,
                NumpyArraySpecType.DISCRETE,
                NumpyArraySpecType.CATEGORICAL
            ):
                while True:
                    sampled_v = self._rng.integers(lb, ub+1, shape, output_spec.dtype)
                    if sampled_v != ret.xs[k]:
                        ret.xs[k] = sampled_v
                        break
        return ret