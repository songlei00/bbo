import unittest

import numpy as np
import torch

from bbo.utils.converters.embedder import (
    OneHotEmbedder,
    DenseEmbedder
)


class OneHotEmbedderTest(unittest.TestCase):
    def setUp(self):
        self.num_feasible_values = [2, 5, 10]

    def test_run_np(self):
        embedder = OneHotEmbedder(self.num_feasible_values)
        data = np.random.randint(
            np.zeros(len(self.num_feasible_values)),
            np.array(self.num_feasible_values),
            (3, len(self.num_feasible_values))
        )
        onehot_data = embedder.map(data)
        data_ = embedder.unmap(onehot_data)
        self.assertTrue((data == data_).all())

    def test_run_torch(self):
        embedder = OneHotEmbedder(self.num_feasible_values)
        data = torch.randint(
            2**63 - 1,
            size=(3, len(self.num_feasible_values))
        ) % torch.tensor(self.num_feasible_values)
        onehot_data = embedder.map(data)
        data_ = embedder.unmap(onehot_data)
        self.assertTrue((data == data_).all())


class DenseEmbedderTest(unittest.TestCase):
    def setUp(self):
        self.num_feasible_values = [2, 5, 10]
        self.d = 2

    def test_run(self):
        embedder = DenseEmbedder(self.num_feasible_values, self.d)
        data = torch.randint(
            2**63 - 1,
            size=(3, len(self.num_feasible_values))
        ) % torch.tensor(self.num_feasible_values)
        dense_data = embedder.map(data)
        self.assertEqual(dense_data.shape, (3, len(self.num_feasible_values)*self.d))

        with self.assertRaises(RuntimeError):
            embedder.unmap(dense_data)