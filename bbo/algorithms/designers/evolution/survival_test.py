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

from bbo.algorithms.designers.evolution import templates
from bbo.algorithms.designers.evolution.survival import AgeBasedSurvival
from bbo.utils.testing import create_dummy_ps

ps, trials, cardinality = create_dummy_ps(5)


def test_age_based_survival():
    converter = templates.DefaultPopulationConverter(ps)
    pop = converter.to_population(trials)
    pop.ages = np.array([3, 2, 1, 5, 4])
    survival = AgeBasedSurvival(3)
    pop = survival(pop)
    assert len(pop) == 3
    assert np.all(pop.ages != 5) and np.all(pop.ages !=4)