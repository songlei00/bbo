import torch.nn as nn
from gpytorch.kernels import Kernel


class WarpKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        wrapper: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.wrapper = wrapper

    def forward(self, x1, x2, **params):
        return self.base_kernel(self.wrapper(x1), self.wrapper(x2), **params)