from abc import ABC, abstractmethod

import numpy as np

from bbo.utils.converters.converter import NumpyArraySpec


class MutationOperator(ABC):
    @abstractmethod
    def __call__(
        self,
        curr_val: np.ndarray,
        spec: NumpyArraySpec,
    ) -> np.ndarray:
        pass
