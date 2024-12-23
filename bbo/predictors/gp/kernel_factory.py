from typing import List, Dict, Optional

from attrs import define, field, validators
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models.kernels import CategoricalKernel

from bbo.predictors.gp.input_wrapper import (
    IdentityWrapper,
    MLPWrapper,
    KumarWrapper,
    WrapperKernel,
)
from bbo.predictors.gp.kernels import MixedKernel
from bbo.utils.converters.converter import SpecType


@define
class KernelFactory:
    _kernel_type: str = field(
        default='matern52',
        validator=validators.in_(['matern52', 'mlp', 'kumar', 'mixed'])
    )
    _ard_dims: Optional[int] = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
        kw_only=True
    )
    _hidden_features: List | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(list)),
        kw_only=True
    )
    _type2num: Dict | None = field(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(
                validators.instance_of(SpecType),
                validators.instance_of(int)
            )
        ),
        kw_only=True
    )

    def __call__(self):
        if self._kernel_type in ['matern52', 'mlp', 'kumar']:
            if self._kernel_type == 'matern52':
                wrapper = IdentityWrapper()
            elif self._kernel_type == 'mlp':
                wrapper = MLPWrapper(self._hidden_features, activate_final=True)
            elif self._kernel_type == 'kumar':
                wrapper = KumarWrapper()
            else:
                raise NotImplementedError('Unsupported kernel wrapper')
            base_kernel = ScaleKernel(
                base_kernel=MaternKernel(
                    nu=2.5,
                    ard_num_dims=self._ard_dims,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            covar_module = WrapperKernel(base_kernel, wrapper)
        elif self._kernel_type == 'mixed':
            double_kernel = ScaleKernel(MaternKernel())
            cat_kernel = CategoricalKernel()
            covar_module = MixedKernel(
                double_kernel, cat_kernel,
                self._type2num[SpecType.DOUBLE],
                self._type2num[SpecType.CATEGORICAL],
                mix_mode='add'
            )
        else:
            raise NotImplementedError('Unsupported kernel type')

        return covar_module