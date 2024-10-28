from gpytorch.means import ConstantMean

from bbo.algorithms.bo_utils.input_wrapper import (
    MLPWrapper,
    WrapperMean,
)


def mean_factory(mean_type, mean_config=None):
    mean_type = mean_type or 'constant'
    if mean_type == 'constant':
        mean_module = ConstantMean()
    elif mean_type == 'mlp':
        wrapper = MLPWrapper(mean_config['hidden_features'])
        mean_module = WrapperMean(wrapper)
    else:
        raise NotImplementedError
    return mean_module