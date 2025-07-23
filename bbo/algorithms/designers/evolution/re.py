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

from bbo.shared.base_study_config import ProblemStatement
from bbo.algorithms.designers.evolution import templates
from bbo.algorithms.designers.evolution.sampling import RandomSampler
from bbo.algorithms.designers.evolution.selection import TournamentSelection
from bbo.algorithms.designers.evolution.mutation import RandomMutation
from bbo.algorithms.designers.evolution.survival import AgeBasedSurvival
from bbo.utils import get_rng


class RegularizedEvolutionDesigner(templates.CanonicalEvolutionDesigner):
    def __init__(
        self,
        problem_statement: ProblemStatement,
        pop_size: int = 25,
        tournament_size: int = 5,
        num_mutation: int = 1,
        seed_or_rng: int | None = None
    ):
        self._problem_statement = problem_statement
        self._pop_size = pop_size
        self._tournament_size = tournament_size
        self._num_mutation = num_mutation
        self._rng = get_rng(seed_or_rng)
        self._converter = templates.DefaultPopulationConverter(self._problem_statement)
        super().__init__(
            problem_statement=self._problem_statement,
            pop_size=self._pop_size,
            offspring_size=1,
            sampling=RandomSampler(self._converter, self._rng),
            selection=TournamentSelection(
                self._tournament_size,
                self._problem_statement.metric_information_item().goal,
                self._rng
            ),
            crossover=None,
            mutation=RandomMutation(self._converter, rng=self._rng, k=self._num_mutation),
            survival=AgeBasedSurvival(self._pop_size, is_ordered=True)
        )