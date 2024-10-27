from abc import ABCMeta, abstractmethod
from typing import Sequence, List

import numpy as np

from bbo.utils.metric_config import MetricInformation
from bbo.utils.trial import ParameterValue, Metric, ParameterDict, MetricDict, Trial


class BaseInputConverter(metaclass=ABCMeta):
    @abstractmethod
    def convert(self, trials: Sequence[Trial]) -> np.ndarray:
        """Convert trials to a ndarray with shape (number of trials, number of features)"""
        pass

    @abstractmethod
    def to_parameter_values(self, array: np.ndarray) -> List[ParameterValue]:
        """Convert a ndarray to parameter values for trial
        
        This can be considered as a reverse transform of convert
        """
        pass

    @property
    @abstractmethod
    def output_spec(self):
        """Return the spec after transform"""
        pass
    
    @property
    @abstractmethod
    def parameter_config(self):
        """Return the original parameter config"""
        pass


class BaseOutputConverter(metaclass=ABCMeta):
    @abstractmethod
    def convert(self, trials: Sequence[Trial]) -> np.ndarray:
        pass

    @abstractmethod
    def to_metrics(self, array: np.ndarray) -> List[Metric]:
        pass

    @property
    @abstractmethod
    def metric_information(self) -> MetricInformation:
        pass


class BaseTrialConverter(metaclass=ABCMeta):
    @abstractmethod
    def convert(self, trials: Sequence[Trial]):
        pass

    @abstractmethod
    def to_features(self, trials: Sequence[Trial]):
        pass

    @abstractmethod
    def to_labels(self, trials: Sequence[Trial]):
        pass

    @abstractmethod
    def to_trials(self, features, labels) -> Sequence[ParameterDict]:
        pass

    @abstractmethod
    def to_parameters(self, features) -> Sequence[Trial]:
        pass

    @abstractmethod
    def to_metrics(self, labels) -> List[MetricDict]:
        pass