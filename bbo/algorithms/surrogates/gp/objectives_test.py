import torch

from bbo.algorithms.surrogates.gp.objectives import neg_log_marginal_likelihood
from bbo.algorithms.surrogates.gp.means import ConstantMean
from bbo.algorithms.surrogates.gp.kernels import RBFKernel


def test_neg_log_marginal_likelihood_1():
    X = torch.randn((20, 3))
    y = torch.randn(20, 1)
    mean_module = ConstantMean()
    cov_module = RBFKernel()
    loss = neg_log_marginal_likelihood(X, y, mean_module, cov_module)
    assert loss.numel() == 1