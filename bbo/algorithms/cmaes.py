from collections import deque
from typing import Optional, Sequence
from attrs import define, field, validators

import cma
import numpy as np

from bbo.algorithms.base import Designer
from bbo.algorithms.random import RandomDesigner
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.converters.converter import ArrayTrialConverter
from bbo.utils.parameter_config import ParameterType
from bbo.utils.metric_config import ObjectiveMetricGoal
from bbo.utils.trial import Trial


@define
class CMAESConfig:
    population_size: int = 20
    sigma: float = 0.01


@define
class CMAESDesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement),
    )
    _config: CMAESConfig = field(
        factory=CMAESConfig,
        validator=validators.instance_of(CMAESConfig),
    )
    _seed: Optional[int] = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
    )

    def __attrs_post_init__(self):
        if not all([
            pc.type == ParameterType.DOUBLE
            for pc in self._problem_statement.search_space.parameters
        ]):
            raise RuntimeError('CMAES can only tackle DOUBLE parameters')

        self._is_flip = (self._problem_statement.objective.metrics[0].goal ==
                         ObjectiveMetricGoal.MAXIMIZE)
        self._init_designer = RandomDesigner(self._problem_statement)
        self._converter = ArrayTrialConverter.from_problem(self._problem_statement)

        trial0 = self._init_designer.suggest(count=1)
        x0 = self._converter.to_features(trial0)[0]
        lb, ub = [], []
        for spec in self._converter.output_spec.values():
            lb.append(spec.bounds[0])
            ub.append(spec.bounds[1])
        self._cmaes = cma.CMAEvolutionStrategy(
            x0, self._config.sigma, {
                'popsize': self._config.population_size,
                'bounds': (lb, ub),
                'seed': self._seed,
            },
        )
        self._curr_trials = deque(maxlen=self._config.population_size)

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        count = count or 1
        if count != 1:
            raise ValueError('CMAES can only suggest one suggestion now')

        pop = np.asarray(self._cmaes.ask(count))
        
        return self._converter.to_trials(pop)

    def update(self, completed: Sequence[Trial]) -> None:
        while completed:
            self._curr_trials.append(completed.pop())
            
            if len(self._curr_trials) == self._config.population_size:
                features, labels = self._converter.convert(self._curr_trials)
                if self._is_flip:
                    labels = - labels
                labels = [v.item() for v in labels]
                self._cmaes.tell(features, labels)
                self._curr_trials.clear()