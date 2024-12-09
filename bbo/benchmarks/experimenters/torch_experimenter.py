from typing import List, Callable

from torch import Tensor

from bbo.benchmarks.experimenters.base import BaseExperimenter
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.converters.torch_converter import TorchArrayTrialConverter
from bbo.utils.trial import Trial, check_bounds


class TorchExperimenter(BaseExperimenter):
    def __init__(
        self,
        impl: Callable[[Tensor], Tensor],
        problem_statement: ProblemStatement,
    ):
        self._dim = problem_statement.search_space.num_parameters()
        self._impl = impl
        self._problem_statement = problem_statement

        self._converter = TorchArrayTrialConverter.from_problem(problem_statement, scale=False)

    def evaluate(self, suggestions: List[Trial]):
        for trial in suggestions:
            check_bounds(self._problem_statement.search_space, trial)
        features = self._converter.to_features(suggestions)
        m = self._impl(features)
        metrics = self._converter.to_metrics(m)
        for suggestion, m in zip(suggestions, metrics):
            suggestion.complete(m)
        return suggestions
    
    def problem_statement(self) -> ProblemStatement:
        return self._problem_statement
