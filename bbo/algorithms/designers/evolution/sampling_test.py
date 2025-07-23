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
from bbo.algorithms.designers.evolution.sampling import RandomSampler
from bbo.utils.testing import create_dummy_ps

ps, trials, cardinality = create_dummy_ps(5)


def test_random_sampler():
    rng = np.random.default_rng()
    converter = templates.DefaultPopulationConverter(ps)
    sampler = RandomSampler(converter, rng)
    pop = sampler(3)
    assert len(pop) == 3