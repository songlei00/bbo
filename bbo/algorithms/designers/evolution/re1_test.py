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

from bbo.shared.base_study_config import ObjectiveMetricGoal
from bbo.algorithms.converters.core import DefaultTrialConverter
from bbo.algorithms.designers.evolution.re1 import (
    tournament_selection,
    random_mutation
)
from bbo.utils.testing import create_dummy_ps

ps, trials, cardinality = create_dummy_ps(5)


@pytest.mark.parametrize('tournament_size', [1, 3, 5])
def test_tournament_selection(tournament_size):
    rng = np.random.default_rng(0)
    winner = tournament_selection(trials, tournament_size, ObjectiveMetricGoal.MAXIMIZE, rng)
    def get_trial_y(trial):
        return list(trial.final_measurement.metrics.values())[0].value
    assert sum(get_trial_y(winner) > get_trial_y(t) for t in trials) >= tournament_size - 1


@pytest.mark.parametrize('k', [1, 3])
def test_random_mutation(k):
    rng = np.random.default_rng()
    converter = DefaultTrialConverter.from_study_config(ps)
    trial = random_mutation(trials[0], converter, k, rng)
    x1 = converter.to_features([trials[0]])
    x2 = converter.to_features([trial])
    assert sum([x1[k] != x2[k] for k in x1.keys()]) == k