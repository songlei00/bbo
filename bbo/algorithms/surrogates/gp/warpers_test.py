import torch

from bbo.algorithms.surrogates.gp.warpers import MLPWarp, KumarWarp


def test_mlp():
    warp_module = MLPWarp(5, [32, 3])
    X = torch.randn(10, 5)
    Y = warp_module(X)
    assert Y.shape == (10, 3)


def test_kumar():
    warp_module = KumarWarp()
    assert len(list(warp_module.parameters())) == 2
    X = torch.rand(10, 5)
    Y = warp_module(X)
    assert Y.shape == X.shape