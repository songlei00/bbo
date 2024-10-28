import numpy as np


def dummy(array: np.ndarray, n_obj=1):
    return np.random.rand(array.shape[0], n_obj)
