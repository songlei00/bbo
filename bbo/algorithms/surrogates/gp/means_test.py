import torch
from bbo.algorithms.surrogates.gp.means import ConstantMean


def test_constant_mean():
    mean = ConstantMean()
    X = torch.randn(10, 2)
    m = mean(X)
    assert m.shape == (10, 1)
    assert torch.all(m == m[0])