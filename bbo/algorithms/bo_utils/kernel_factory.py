from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models.kernels import CategoricalKernel

from bbo.algorithms.bo_utils.input_wrapper import (
    IdentityWrapper,
    MLPWrapper,
    KumarWrapper,
    WrapperKernel,
)
from bbo.algorithms.bo_utils.kernels import MixedKernel
from bbo.utils.converters.converter import SpecType


def kernel_factory(kernel_type, kernel_config=None):
    kernel_type = kernel_type or 'matern52'

    if kernel_type in ['matern52', 'mlp', 'kumar']:
        if kernel_type == 'matern52':
            wrapper = IdentityWrapper()
        elif kernel_type == 'mlp':
            wrapper = MLPWrapper(kernel_config['hidden_features'], activate_final=True)
        elif kernel_type == 'kumar':
            wrapper = KumarWrapper()
        else:
            raise NotImplementedError('Unsupported kernel wrapper')
        base_kernel = ScaleKernel(MaternKernel())
        covar_module = WrapperKernel(base_kernel, wrapper)
    elif kernel_type == 'mixed':
        double_kernel = ScaleKernel(MaternKernel())
        cat_kernel = CategoricalKernel()
        covar_module = MixedKernel(
            double_kernel, cat_kernel,
            kernel_config['type2num'][SpecType.DOUBLE],
            kernel_config['type2num'][SpecType.CATEGORICAL],
            mix_mode='add'
        )
    else:
        raise NotImplementedError('Unsupported kernel type')

    return covar_module