import abc
from typing import List

from bbo.utils.trial import Trial
from bbo.utils.problem_statement import ProblemStatement


class BaseExperimenter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, suggestions: List[Trial]):
        pass
    
    @abc.abstractmethod
    def problem_statement(self) -> ProblemStatement:
        pass