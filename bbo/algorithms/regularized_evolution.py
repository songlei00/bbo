from typing import Optional, Sequence
from collections import deque

import numpy as np
from attrs import define, field, validators

from bbo.algorithms.base import Designer
from bbo.algorithms.random import RandomDesigner
from bbo.algorithms.heuristic_utils.base_operator import MutationOperator
from bbo.algorithms.heuristic_utils.mutate_operator import RandomMutation
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.converters.converter import DefaultTrialConverter
from bbo.utils.metric_config import ObjectiveMetricGoal
from bbo.utils.trial import Trial


@define
class RegularizedEvolutionDesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement),
    )
    _mutate_operator: MutationOperator = field(
        factory=RandomMutation,
        validator=validators.instance_of(MutationOperator),
    )
    _population_size: int = field(
        default=25,
        validator=validators.instance_of(int),
    )
    _tournament_size: int = field(
        default=5,
        validator=validators.instance_of(int),
    )

    def __attrs_post_init__(self):
        self._init_designer = RandomDesigner(self._problem_statement)
        self._converter = DefaultTrialConverter.from_problem(self._problem_statement)
        self._population = deque(maxlen=self._population_size)

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        if len(self._population) < self._population_size:
            ret = self._init_designer.suggest(count)
        else:
            count = count or 1
            ret = []
            for _ in range(count):
                parent = self._tournament()
                child = self._mutate_one(parent)
                ret.append(child)
        return ret

    def update(self, completed: Sequence[Trial]) -> None:
        self._population.extend(completed)

    def _tournament(self) -> Trial:
        idx = np.random.choice(self._population_size, self._tournament_size, replace=False)
        candidates = [self._population[i] for i in idx]
        ys = [list(cand.metrics.values())[0] for cand in candidates]
        goal = self._problem_statement.objective.metrics[0]
        if goal == ObjectiveMetricGoal.MAXIMIZE:
            i = np.argmax(ys)
        else:
            i = np.argmin(ys)
        winner = candidates[i]
        return winner

    def _mutate_one(self, trial: Trial) -> Trial:
        sample = self._converter.to_features(trial)
        mutate_keys = np.random.choice(list(self._converter.output_spec), 1, replace=False)
        for key in mutate_keys:
            spec = self._converter.output_spec[key]
            new_v = self._mutate_operator(sample[key], spec)
            sample[key] = new_v
        return self._converter.to_trials(sample)[0]
