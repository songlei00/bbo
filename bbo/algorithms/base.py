from abc import abstractmethod
from typing import Sequence, Optional, List

from attrs import define, field, validators

from bbo.utils.trial import Trial, check_bounds


@define
class Designer:
    _base_trials: List[Trial] = field(factory=list, init=False)
    _base_epoch: int = field(default=0, init=False)

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        suggestion = self._suggest(count)
        for trial in suggestion:
            check_bounds(self._problem_statement.search_space, trial)
        return suggestion

    def update(self, completed: Sequence[Trial]) -> None:
        for trial in completed:
            check_bounds(self._problem_statement.search_space, trial)
        self._base_trials.extend(completed)
        self._base_epoch += 1
        self._update(completed)

    @property
    def base_trials(self):
        return self._base_trials
    
    @property
    def base_epoch(self):
        return self._base_epoch

    @abstractmethod
    def _suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        pass

    @abstractmethod
    def _update(self, completed: Sequence[Trial]) -> None:
        pass
