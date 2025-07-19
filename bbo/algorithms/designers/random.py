from typing import Optional, Sequence

from attrs import define, field, validators
import numpy as np

from bbo.algorithms.abstractions import Designer, CompletedTrials, ActiveTrials
from bbo.shared.base_study_config import ProblemStatement
from bbo.shared.trial import Trial


@define
class RandomDesigner(Designer):
    problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    dtype: np.dtype = field(default=np.float32)

    def suggest(self, count: Optional[int] = None) -> Sequence[Trial]:
        count = count or 1
        suggestions = []
        for _ in range(count):
            sample = self.problem_statement.search_space.sample()
            suggestions.append(Trial(parameters=sample))
        return suggestions

    def update(self, completed_trials: CompletedTrials, active_trials: ActiveTrials):
        pass