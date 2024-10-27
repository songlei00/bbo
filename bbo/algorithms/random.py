from typing import Optional, Sequence

import numpy as np
from attrs import define, field, validators

from bbo.algorithms.base import Designer
from bbo.utils.converters.converter import SpecType, DefaultTrialConverter
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.trial import Trial


@define
class RandomDesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement),
    )
    _seed: Optional[int] = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
    ) # TODO: use seed to control the behavior

    def __attrs_post_init__(self):
        self._converter = DefaultTrialConverter.from_problem(self._problem_statement)
        # self._rng = np.random.RandomState(self._seed)

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        count = count or 1
        sample = dict()
        for name, spec in self._converter.output_spec.items():
            lb, ub = spec.bounds
            if spec.type == SpecType.DOUBLE:
                sample[name] = np.random.rand(count, 1) * (ub - lb) + lb
            elif spec.type in (
                SpecType.CATEGORICAL,
                SpecType.DISCRETE,
                SpecType.INTEGER,
            ):
                sample[name] = np.random.randint(lb, ub+1, (count, 1))
            else:
                raise ValueError('Unsupported type: {}'.format(spec.type))

        return self._converter.to_trials(sample)

    def update(self, completed: Sequence[Trial]) -> None:
        pass
    