from typing import List

from attrs import define, field, validators
from gpytorch.means import ConstantMean

from bbo.algorithms.bo_utils.input_wrapper import (
    MLPWrapper,
    WrapperMean,
)


@define
class MeanFactory:
    _mean_type: str = field(default='constant', validator=validators.in_(['constant', 'mlp']))
    _hidden_features: List | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(list)),
        kw_only=True
    )

    def __call__(self):
        if self._mean_type == 'constant':
            mean_module = ConstantMean()
        elif self._mean_type == 'mlp':
            wrapper = MLPWrapper(self._hidden_features)
            mean_module = WrapperMean(wrapper)
        else:
            raise NotImplementedError
        return mean_module