import torch

from bbo.algorithms.surrogates.gp.gp import GP

bs, n1, n2, d = 4, 20, 10, 3
X = torch.rand((n1, d))
Y = torch.randn((n1, 1))
gp = GP(X, Y, epochs=30)


class TestGP:
    def test_train(self):
        gp.train()

    def test_predict_2d(self):
        n_query = 10
        query_X = torch.randn((n_query, d))
        mu, var = gp.predict(query_X)
        assert mu.shape == (n_query, 1)
        assert var.shape == (n_query, n_query)

    def test_predict_3d(self):
        n_query = 10
        query_X = torch.randn((bs, n_query, d))
        vmap_predict = torch.vmap(gp.predict)
        mu, var = vmap_predict(query_X)
        assert mu.shape == (bs, n_query, 1)
        assert var.shape == (bs, n_query, n_query)