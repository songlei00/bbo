from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors.torch_priors import GammaPrior

from bbo.algorithms.bo_utils.input_wrapper import (
    IdentityWrapper,
    MLPWrapper,
    KumarWrapper,
    WrapperKernel,
)


def kernel_factory(kernel_type, kernel_config=None):
    kernel_type = kernel_type or 'matern52'
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

    return covar_module