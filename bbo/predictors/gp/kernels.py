from typing import Dict

import torch
from torch import Tensor
from torch import nn
from gpytorch.kernels import Kernel

from bbo.utils.converters.converter import SpecType


class MixedKernel(Kernel):
    _support_type = (
        SpecType.DOUBLE,
        SpecType.CATEGORICAL
    )
    
    def __init__(
        self,
        double_kernel: Kernel,
        cat_kernel: Kernel,
        num_double: int,
        num_cat: int,
        mix_mode: str = 'add',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.type2kernel = nn.ModuleDict({
            SpecType.DOUBLE.name: double_kernel,
            SpecType.CATEGORICAL.name: cat_kernel,
        })
        self.num_double = num_double
        self.num_cat = num_cat
        self.mix_mode = mix_mode
        self.transform = nn.Softplus()
        self.mix_weight = nn.Parameter(torch.ones(1)*0.5)

    def forward(self, x1, x2, **params):
        # split by variable type
        double_x1, cat_x1, rest_x1 = torch.tensor_split(
            x1, (self.num_double, self.num_double+self.num_cat), dim=-1
        )
        double_x2, cat_x2, rest_x2 = torch.tensor_split(
            x2, (self.num_double, self.num_double+self.num_cat), dim=-1
        )
        assert rest_x1.shape[-1] == 0 and rest_x2.shape[-1] == 0

        # calculate kernel matrix for different variable types
        type2K = dict()
        type2K[SpecType.DOUBLE] = self.type2kernel[SpecType.DOUBLE.name](
            double_x1, double_x2, **params
        )
        type2K[SpecType.CATEGORICAL] = self.type2kernel[SpecType.CATEGORICAL.name](
            cat_x1, cat_x2, **params
        )

        # mix different kernel matrix
        if self.mix_mode == 'add':
            K = type2K[SpecType.DOUBLE] + type2K[SpecType.CATEGORICAL]
            return K
        elif self.mix_mode == 'mul':
            K = type2K[SpecType.DOUBLE] * type2K[SpecType.CATEGORICAL]
            return K
        elif self.mix_mode == 'both':
            add_K = type2K[SpecType.DOUBLE] + type2K[SpecType.CATEGORICAL]
            mul_K = type2K[SpecType.DOUBLE] * type2K[SpecType.CATEGORICAL]
            mix_weight = self.transform(self.mix_weight)
            K = (1 - mix_weight) * add_K + mix_weight * mul_K
            return K
        else:
            raise ValueError('Only add, mul, both are supported')