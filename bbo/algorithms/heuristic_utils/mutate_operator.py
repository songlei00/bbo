import numpy as np

from bbo.algorithms.heuristic_utils.base_operator import MutationOperator
from bbo.utils.converters.converter import (
    NumpyArraySpec,
    SpecType,
)


class RandomMutation(MutationOperator):
    def __call__(
        self,
        curr_val: np.ndarray,
        spec: NumpyArraySpec,
    ) -> np.ndarray:
        lb, ub = spec.bounds
        shape = (1, 1)
        if spec.type == SpecType.DOUBLE:
            v = np.random.rand(*shape) * (ub - lb) + lb
        elif spec.type in (
            SpecType.CATEGORICAL,
            SpecType.DISCRETE,
            SpecType.INTEGER,
        ):
            v = np.random.randint(lb, ub+1, shape)
        else:
            raise NotImplementedError('Unsupported type: {}'.format(spec.type))
        return v


class PerturbMutation(MutationOperator):
    def __call__(
        self,
        curr_val: np.ndarray,
        spec: NumpyArraySpec
    ) -> np.ndarray:
        lb, ub = spec.bounds
        shape = (1, 1)
        if spec.type == SpecType.DOUBLE:
            v = curr_val.value + np.random.randn(*shape)
            v = np.clip(v, lb, ub)
        elif spec.type in (
            SpecType.INTEGER,
            SpecType.DISCRETE,
        ):
            v = curr_val.value + np.random.choice([-1, 1], shape)
            v = np.clip(v, lb, ub)
        elif spec.type == SpecType.CATEGORICAL:
            v = np.random.randint(lb, ub+1, shape)
        else:
            raise NotImplementedError('Unsupported type: {}'.format(spec.type))
        return v
