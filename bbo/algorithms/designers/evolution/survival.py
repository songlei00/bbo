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


@define
class AgeBasedSurvival(templates.Survival):
    _target_size: int = field(validator=validators.instance_of(int))

    def __call__(self, population: templates.Population) -> templates.Population:
        ages = population.ages
        indices = np.argsort(ages)[: self._target_size]
        return population[indices]