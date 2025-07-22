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

from bbo.algorithms.designers.evolution import templates
from bbo.algorithms.designers.evolution.selection import (
    RandomSelection,
    TournamentSelection
)
from bbo.shared.base_study_config import ObjectiveMetricGoal
from bbo.utils.testing import create_dummy_ps

ps, trials, _ = create_dummy_ps(5)
converter = templates.DefaultPopulationConverter(ps)
population = converter.to_population(trials)


@pytest.mark.parametrize('size', [1, 3])
def test_random_selection(size):
    selection = RandomSelection()
    selected = selection(population, size)
    assert len(selected) == size


@pytest.mark.parametrize('tournament_size', [1, 3, 5])
def test_tournament_selection(tournament_size):
    selection = TournamentSelection(tournament_size, ObjectiveMetricGoal.MAXIMIZE)
    selected = selection(population)
    assert len(selected) == 1
    selected_y = selected.y_item().item()
    all_ys = [ind.y_item() for ind in population]
    assert sum(selected_y > y for y in all_ys) >= tournament_size - 1