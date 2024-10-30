from typing import Optional, Sequence

import numpy as np
from attrs import define, field, validators

from bbo.algorithms.base import Designer
from bbo.algorithms.sampling.random import RandomDesigner
from bbo.algorithms.heuristic_utils.base_operator import MutationOperator
from bbo.algorithms.heuristic_utils.mutate_operator import RandomMutation
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.converters.converter import DefaultTrialConverter
from bbo.utils.trial import Trial, is_better_than


@define
class LocalSearchDesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement),
    )
    _mutate_operator: MutationOperator = field(
        factory=RandomMutation,
        validator=validators.instance_of(MutationOperator),
    )

    def __attrs_post_init__(self):
        self._init_designer = RandomDesigner(self._problem_statement)
        self._converter = DefaultTrialConverter.from_problem(self._problem_statement)
        self._best_suggestion = None

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        if self._best_suggestion is None:
            ret = self._init_designer.suggest(count)
        else:
            count = count or 1
            ret = [self._suggest_one() for _ in range(count)]
        return ret

    def update(self, completed: Sequence[Trial]) -> None:
        for suggestion in completed:
            if self._best_suggestion is None or is_better_than(
                self._problem_statement.objective,
                suggestion,
                self._best_suggestion
            ):
                self._best_suggestion = suggestion
            else:
                # TODO: accept by probability
                pass

    def _suggest_one(self) -> Trial:
        sample = self._converter.to_features([self._best_suggestion])
        mutate_keys = np.random.choice(list(self._converter.output_spec), 1, replace=False)
        for key in mutate_keys:
            spec = self._converter.output_spec[key]
            new_v = self._mutate_operator(sample[key], spec)
            sample[key] = new_v
        return self._converter.to_trials(sample)[0]
