import abc
from typing import List, Tuple, Sequence
from attrs import define, field, validators

import numpy as np

from bbo.shared.base_study_config import ObjectiveMetricGoal, MetricInformation, ProblemStatement
from bbo.shared.trial import Trial
from bbo.benchmarks.experimenters.experimenter import Experimenter
from bbo.benchmarks.experimenters.numpy_experimenter import NumpyExperimenter


class SyntheticFunction(abc.ABC):
    dim: int
    bounds: List[Tuple[float, float]]
    goal: ObjectiveMetricGoal
    optimal_value: float | None
    optimal_points: List[Tuple[float]] | None
    num_objectives: int = 1

    @abc.abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        pass


@define
class SyntheticExperimenter(Experimenter):
    _func_impl: SyntheticFunction = field(validator=validators.instance_of(SyntheticFunction))
    _metric_name: str = field(default='y', validator=validators.instance_of(str))

    def __attrs_post_init__(self):
        self._impl = NumpyExperimenter(self._func_impl, self.problem_statement())

    def evaluate(self, trials: Sequence[Trial]):
        return self._impl.evaluate(trials)

    def problem_statement(self) -> ProblemStatement:
        ps = ProblemStatement()
        for i, (lb, ub) in enumerate(self._func_impl.bounds):
            ps.search_space.add_float_param(f'x{i}', lb, ub)
        ps.metric_information.append(
            MetricInformation(self._metric_name, self._func_impl.goal)
        )
        return ps


class Branin2D(SyntheticFunction):
    """https://www.sfu.ca/~ssurjano/branin.html"""

    dim = 2
    bounds = [
        (-5, 10),
        (0, 15)
    ]
    goal = ObjectiveMetricGoal.MINIMIZE
    optimal_value = 0.397887
    optimal_points = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        x1 = X[..., 0]
        x2 = X[..., 1]

        y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return y
    

class Ackley(SyntheticFunction):
    goal = ObjectiveMetricGoal.MINIMIZE
    optimal_value = 0
    
    def __init__(
        self,
        dim,
        bounds: List[Tuple[float, float]] | None = None
    ):
        self.dim = dim
        self.bounds = bounds if bounds is not None else [(-32.768, 32.768)] * dim
        self.optimal_points = [tuple(0.0 for _ in range(dim))]
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def __call__(self, X: np.ndarray) -> np.ndarray:
        a, b, c = self.a, self.b, self.c
        part1 = -a * np.exp(-np.linalg.norm(X, axis=-1) * b / np.sqrt(self.dim))
        part2 = -(np.exp(np.mean(np.cos(c * X), axis=-1)))
        return part1 + part2 + a + np.exp(1)