from abc import ABC, abstractmethod

import numpy as np

from bbo.utils.converters.converter import FeatureSpec


class MutationOperator(ABC):
    @abstractmethod
    def __call__(
        self,
        curr_val: np.ndarray,
        spec: FeatureSpec,
    ) -> np.ndarray:
        pass
