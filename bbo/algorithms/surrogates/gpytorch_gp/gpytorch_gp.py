import math
from typing import Union

from attrs import define, field, validators
import torch
from gpytorch.models import ExactGP
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import torch_priors

from bbo.algorithms.abstractions import Surrogate
from bbo.algorithms.surrogates.gp.warpers import KumarWarp, MLPWarp
from bbo.algorithms.surrogates.gpytorch_gp.kernels import WarpKernel
from bbo.utils import attrs_utils, logging_utils

logger = logging_utils.get_logger(__name__)

default_priors = {
    'lengthscale_prior': torch_priors.GammaPrior(3.0, 6.0),
    'outputscale_prior': torch_priors.GammaPrior(2.0, 0.15),
    'noise_prior': torch_priors.LogNormalPrior(loc=-4.0, scale=1.0)
}

def default_kernel_factory(
    base_kernel_name: str,
    warp_module_name: str | None,
    dims: int | None
):
    ard_dims = dims
    if ard_dims:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(dims))  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    if warp_module_name is not None:
        if warp_module_name == 'kumar':
            warp_module = KumarWarp()
        elif warp_module_name == 'mlp':
            warp_module = MLPWarp(dims, [32, 16])
            ard_dims = 16
        else:
            raise NotImplementedError(f'warp_module_name={warp_module_name} is not supported')

    if base_kernel_name == 'matern52':
        kernel = ScaleKernel(
            MaternKernel(
                ard_num_dims=ard_dims,
                lengthscale_constraint=lengthscale_constraint,
                lengthscale_prior=default_priors['lengthscale_prior']
            ),
            outputscale_constraint=outputscale_constraint,
            outputscale_prior=default_priors['outputscale_prior']
        )
    elif base_kernel_name == 'rbf':
        kernel = ScaleKernel(
            RBFKernel(
                ard_num_dims=ard_dims,
                lengthscale_constraint=lengthscale_constraint,
                lengthscale_prior=default_priors['lengthscale_prior']
            ),
            outputscale_constraint=outputscale_constraint,
            outputscale_prior=default_priors['outputscale_prior']
        )
    else:
        raise NotImplementedError(f'base_kernel_name={base_kernel_name} is not supported')
    
    if warp_module_name is not None:
        kernel = WarpKernel(kernel, warp_module)

    return kernel


class GPModel(ExactGP):
    def __init__(
        self,
        train_X,
        train_Y,
        kernel,
        likelihood
    ):
        super(GPModel, self).__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


@define
class GPyTorchGP(Surrogate):
    _X: torch.Tensor = field(validator=attrs_utils.assert_2dtensor)
    _Y: torch.Tensor = field(validator=attrs_utils.assert_2dtensor)
    _base_kernel_name: str = field(default='matern52', validator=validators.in_(['matern52', 'rbf']))
    _warp_module_name: str | None = field(default=None, validator=validators.optional(validators.in_(['kumar', 'mlp'])))
    _use_ard: bool = field(default=True)
    _device: str = field(default='cpu', kw_only=True)

    _lr: float = field(default=0.01, converter=float, validator=validators.instance_of(float))
    _epochs: int = field(default=100, converter=int, validator=validators.instance_of(int))

    _gp_model: ExactGP = field(validator=validators.instance_of(ExactGP), init=False)

    def __attrs_post_init__(self):
        if isinstance(self._device, str) and self._device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"Device {self._device} is not available. Use CPU")
        self._device = torch.device(self._device if torch.cuda.is_available() else 'cpu')

    def train(self):
        train_X, train_Y = self._X, self._Y.flatten()
        noise_constraint = Interval(5e-4, 0.2)
        
        likelihood = GaussianLikelihood(
            noise_constraint=noise_constraint,
            noise_prior=default_priors['noise_prior']
        ).to(self._device)
        ard_dims = train_X.shape[-1] if self._use_ard else None
        kernel = default_kernel_factory(self._base_kernel_name, self._warp_module_name, ard_dims)
        model = GPModel(
            train_X=train_X,
            train_Y=train_Y,
            kernel=kernel,
            likelihood=likelihood,
        ).to(self._device)

        model.train()
        likelihood.train()
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # hypers = {}
        # hypers["covar_module.outputscale"] = 1.0
        # hypers["covar_module.base_kernel.lengthscale"] = 0.5
        # hypers["likelihood.noise"] = 0.005
        # model.initialize(**hypers)

        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=self._lr)

        for _ in range(self._epochs):
            optimizer.zero_grad()
            output = model(train_X)
            loss = -mll(output, train_Y)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        self._gp_model = model

    def predict(self, query_X: torch.Tensor):
        dist = self._gp_model.likelihood(self._gp_model(query_X))
        return dist.mean.unsqueeze(-1), dist.covariance_matrix
    