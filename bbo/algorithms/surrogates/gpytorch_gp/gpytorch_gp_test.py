import pytest
import torch

from bbo.algorithms.surrogates.gpytorch_gp.gpytorch_gp import GPyTorchGP

bs, n1, n2, d = 4, 20, 10, 3
X_32 = torch.rand((n1, d), dtype=torch.float32)
Y_32 = torch.randn((n1, 1), dtype=torch.float32)
X_64 = X_32.double()
Y_64 = Y_32.double()

@pytest.mark.parametrize("gp", [
    GPyTorchGP(X_64, Y_64),
])
class TestGPyTorchGP:
    def test_predict_2d(self, gp):
        gp.train()
        n_query = 10
        query_X = torch.randn((n_query, d))
        mu, var = gp.predict(query_X)
        assert mu.shape == (n_query, 1)
        assert var.shape == (n_query, n_query)

    def test_predict_3d(self, gp):
        gp.train()
        n_query = 10
        query_X = torch.randn((bs, n_query, d))
        parallel_pred_mu, parallel_pred_var = gp.predict(query_X)
        assert parallel_pred_mu.shape == (bs, n_query, 1)
        assert parallel_pred_var.shape == (bs, n_query, n_query)

        seq_pred_mu, seq_pred_var = [], []
        for i in range(bs):
            pred_mu, pred_var = gp.predict(query_X[i])
            seq_pred_mu.append(pred_mu.unsqueeze(0))
            seq_pred_var.append(pred_var.unsqueeze(0))
        seq_pred_mu = torch.cat(seq_pred_mu, dim=0)
        seq_pred_var = torch.cat(seq_pred_var, dim=0)
        assert torch.allclose(parallel_pred_mu, seq_pred_mu)
        assert torch.allclose(parallel_pred_var, seq_pred_var)
