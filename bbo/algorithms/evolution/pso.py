from typing import Optional, Sequence, Dict

from attrs import define, field, validators
import numpy as np

from bbo.algorithms.base import Designer
from bbo.algorithms.sampling.random import RandomDesigner
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.converters.converter import SpecType, DefaultTrialConverter, BaseTrialConverter
from bbo.utils.trial import Trial, is_better_than


@define
class PSODesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement),
    )
    _population_size: int = field(
        default=25,
        validator=validators.instance_of(int),
    )
    _w: float = field(default=0.9, validator=validators.instance_of(float))
    _c1: float = field(default=1.4, validator=validators.instance_of(float))
    _c2: float = field(default=1.4, validator=validators.instance_of(float))

    _init_designer: Designer = field(init=False)
    _converter: BaseTrialConverter = field(init=False)
    _v: Dict[str, np.ndarray] = field(init=False)
    _curr_id: int = field(default=0, init=False)
    _population: Sequence[Trial] = field(factory=list, init=False)
    _pbest: Sequence[Trial] = field(factory=list, init=False)
    _gbest: Trial | None = field(default=None, init=False)

    def __attrs_post_init__(self):
        self._init_designer = RandomDesigner(self._problem_statement)
        self._converter = DefaultTrialConverter.from_problem(self._problem_statement)
        self._v = {k: np.random.randn(1, 1) for k in self._problem_statement.search_space.parameter_configs.keys()}

    def _suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        count = count or 1
        if count > 1:
            raise ValueError('Only support count=1')
        if len(self._population) < self._population_size:
            return self._init_designer.suggest(count)
        else:
            gbest_sample = self._converter.to_features([self._gbest])
            pbest_sample = self._converter.to_features([self._pbest[self._curr_id]])
            curr_sample = self._converter.to_features([self._population[self._curr_id]])
            sample = dict()
            for name, spec in self._converter.output_spec.items():
                if spec.type == SpecType.DOUBLE:
                    size = curr_sample[name].shape
                    v = self._w * self._v[name] + \
                        self._c1 * np.random.uniform(size=size) * (pbest_sample[name] - curr_sample[name]) + \
                        self._c2 * np.random.uniform(size=size) * (gbest_sample[name] - curr_sample[name])
                    sample[name] = curr_sample[name] + v
                    lb, ub = spec.bounds
                    sample[name] = np.clip(sample[name], lb, ub)
                else:
                    raise NotImplementedError('Unsupported variable type')
            return self._converter.to_trials(sample)

    def _update(self, completed: Sequence[Trial]) -> None:
        for trial in completed:
            self._update_one(trial)

    def _update_one(self, trial: Trial):
        if len(self._population) < self._population_size:
            self._population.append(trial)
            self._pbest.append(trial)
        else:
            if is_better_than(self._problem_statement.objective, trial, self._population[self._curr_id]):
                self._pbest[self._curr_id] = trial
            self._population[self._curr_id] = trial
        if self._gbest is None or is_better_than(self._problem_statement.objective, trial, self._gbest):
            self._gbest = trial
        self._curr_id = (self._curr_id + 1) % self._population_size

    def result(self) -> Sequence[Trial]:
        return [self._gbest]