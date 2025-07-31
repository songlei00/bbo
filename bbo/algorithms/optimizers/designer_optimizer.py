from typing import Callable, Sequence, Dict, List

from attrs import define, field, validators
import torch

from bbo.algorithms.abstractions import Designer, CompletedTrials
from bbo.algorithms.optimizers.base import GradientFreeOptimizer
from bbo.shared.base_study_config import ProblemStatement
from bbo.shared.trial import Trial, Measurement, get_best_trials


@define
class DesignerAsOptimizer(GradientFreeOptimizer):
    _designer_factory: Callable[[ProblemStatement], Designer] = field()
    _batch_size: int = field(default=1, validator=validators.instance_of(int))
    _num_evaluations: int = field(default=100, validator=validators.instance_of(int))

    def optimize(
        self,
        score_fn: Callable[[Sequence[Trial]], Dict[str, torch.Tensor]],
        problem_statement: ProblemStatement,
        *,
        count: int = 1,
        seed_candidates: Sequence[Trial] = tuple()
     ) -> List[Trial]:
        designer = self._designer_factory(problem_statement)
        if seed_candidates:
            designer.update(CompletedTrials(seed_candidates))
        num_iterations = max(self._num_evaluations // self._batch_size, 1)
        trials = []
        for _ in range(num_iterations):
            suggestions = designer.suggest(self._batch_size)
            scores = score_fn(suggestions)
            for i, suggestion in enumerate(suggestions):
                suggestion.complete(Measurement({
                    k: v[i].item() for k, v in scores.items()
                }))
            completed_trials = CompletedTrials(suggestions)
            designer.update(completed_trials)
            trials.extend(completed_trials.trials)
        best_trials = get_best_trials(trials, problem_statement, count=count)
        return best_trials
