from typing import List, Callable

import numpy as np

from bbo.benchmarks.experimenters.base import BaseExperimenter
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.converters.converter import ArrayTrialConverter
from bbo.utils.trial import Trial


class NumpyExperimenter(BaseExperimenter):
    def __init__(
        self,
        impl: Callable[[np.ndarray], np.ndarray],
        problem_statement: ProblemStatement,
    ):
        self._dim = len(problem_statement.search_space.parameters)
        self._impl = impl
        self._problem_statement = problem_statement

        self._converter = ArrayTrialConverter.from_problem(problem_statement, scale=False)

    def evaluate(self, suggestions: List[Trial]):
        features = self._converter.to_features(suggestions)
        m = self._impl(features)
        metrics = self._converter.to_metrics(m)
        for suggestion, m in zip(suggestions, metrics):
            suggestion.complete(m)
        return suggestions

    def problem_statement(self) -> ProblemStatement:
        return self._problem_statement