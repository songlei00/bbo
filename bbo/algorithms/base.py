from abc import abstractmethod
from typing import Sequence, Optional, List

from attrs import define, field, validators, evolve

from bbo.utils.trial import Trial


@define
class Designer:
    _trials: List[Trial] = field(factory=list, init=False)
    _epoch: int = field(default=0, init=False)

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        return self._suggest(count)

    def update(self, completed: Sequence[Trial]) -> None:
        self._trials.extend(completed)
        self._epoch += 1
        self._update(completed)

    def reset(self, trials: Sequence[Trial]=None):
        self._trials = trials or list()
        self._epoch = 0
        self._reset(trials)

    @property
    def trials(self):
        return self._trials

    @abstractmethod
    def _suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        pass

    @abstractmethod
    def _update(self, completed: Sequence[Trial]) -> None:
        pass

    @abstractmethod
    def _reset(self, trials: Sequence[Trial]=None):
        pass
