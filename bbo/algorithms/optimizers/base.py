import abc
from typing import List, Sequence

from bbo.shared.base_study_config import ProblemStatement
from bbo.shared.trial import Trial


class GradientFreeOptimizer(abc.ABC):
    @abc.abstractmethod
    def optimize(
        self,
        score_fn,
        problem_statement: ProblemStatement,
        *,
        count: int = 1,
        seed_candidates: Sequence[Trial] = tuple()
    ) -> List[Trial]:
        pass
