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

from bbo.algorithms.designers.evolution import templates
from bbo.utils import testing

ps, trials, _ = testing.create_dummy_ps(5)
active_trials = testing.generate_trials(ps, 5, False)


def _generate_population(size):
    pop = templates.Population(
        xs = {
            'x1': np.random.rand(size, 1),
            'x2': np.random.rand(size, 1),
        },
        ys = {'obj': np.random.rand(size, 1)},
        ages = np.random.randint(0, 100, size),
        generations = np.random.randint(0, 100, size),
        ids = np.random.randint(0, 100, size),
    )
    return pop


class TestPopulation:
    @pytest.mark.parametrize('indices', [
        3,
        slice(3),
        [2, 3]
    ])
    def test_getitem(self, indices):
        pop = _generate_population(5)
        assert len(pop) == 5
        selected = pop[indices]
        if isinstance(indices, int):
            selected_idx = [indices]
        elif isinstance(indices, slice):
            selected_idx = list(range(*indices.indices(len(pop))))
        else:
            selected_idx = indices
        assert len(selected) == len(selected_idx)
        for individual, idx in zip(selected, selected_idx):
            assert individual == pop[idx]

    def test_add(self):
        pop1 = _generate_population(3)
        pop2 = _generate_population(2)
        pop3 = pop1 + pop2
        assert len(pop3) == 5
        assert pop3[:3] == pop1
        assert pop3[3:] == pop2

    def test_serializable(self):
        pop = _generate_population(3)
        metadata = pop.dump()
        recovered = templates.Population.recover(metadata)
        assert pop == recovered


class TestDefaultPopulationConverter:
    def test_default(self):
        converter = templates.DefaultPopulationConverter(ps)
        pop = converter.to_population(trials)
        recovered = converter.to_trials(pop)
        testing.compare_trials(trials, recovered)

    def test_without_y(self):
        converter = templates.DefaultPopulationConverter(ps)
        pop = converter.to_population(active_trials)
        recovered = converter.to_trials(pop)
        testing.compare_trials(active_trials, recovered)