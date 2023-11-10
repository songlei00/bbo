from abc import ABC, abstractmethod
from typing import Sequence, Optional

from bbo.utils.trial import Trial


class Designer(ABC):
    @abstractmethod
    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        pass

    @abstractmethod
    def update(self, completed: Sequence[Trial]) -> None:
        pass
