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


@define
class RandomSampler(templates.Sampler):
    _converter: templates.PopulationConverter = field(
        validator=validators.instance_of(templates.PopulationConverter)
    )

    def __call__(self, count: int) -> templates.Population:
        output_specs = self._converter.trial_converter.output_specs
        xs = dict()
        for name, output_spec in output_specs.items():
            lb, ub = output_spec.bounds
            shape = (count, 1)
            if output_spec.type == NumpyArraySpecType.DOUBLE:
                xs[name] = np.random.rand(*shape) * (ub - lb) + lb
            elif output_spec.type in (
                NumpyArraySpecType.INTEGER,
                NumpyArraySpecType.DISCRETE,
                NumpyArraySpecType.CATEGORICAL
            ):
                xs[name] = np.random.randint(lb, ub+1, shape)
        return templates.Population(xs)