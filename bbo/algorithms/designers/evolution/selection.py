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

from typing import Optional

import numpy as np
from attrs import define, field, validators

from bbo.algorithms.designers.evolution import templates
from bbo.shared.base_study_config import ObjectiveMetricGoal


@define
class RandomSelection(templates.Selection):
    _rng: np.random.Generator = field(validator=validators.instance_of(np.random.Generator))

    def __call__(self, population: templates.Population, count: Optional[int] = None) -> templates.Population:
        count = count or 1
        indices = self._rng.choice(len(population), size=count, replace=False)
        return population[indices]


@define
class TournamentSelection(templates.Selection):
    _tournament_size: int = field(validator=validators.instance_of(int))
    _goal: ObjectiveMetricGoal = field(validator=validators.instance_of(ObjectiveMetricGoal))
    _rng: np.random.Generator = field(validator=validators.instance_of(np.random.Generator))

    def __call__(self, population: templates.Population) -> templates.Population:
        indices = self._rng.choice(len(population), size=self._tournament_size, replace=False)
        candidates = population[indices]
        ys = candidates.y_item()
        if self._goal.is_maximize:
            ind = np.argmax(ys).item()
        else:
            ind = np.argmin(ys).item()
        winner = candidates[ind]
        return winner