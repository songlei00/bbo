import pytest
import numpy as np

from bbo.utils import math_utils


@pytest.mark.parametrize(
    'func, data, expected',
    [
        (math_utils.argmin, [1, 2, 3], 0),
        (math_utils.argmax, [1, 2, 3], 2),
    ]
)
def test_operator(func, data, expected):
    assert func(data) == expected


@pytest.mark.parametrize(
    'd1, d2, expected',
    [
        ({'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, True),
        ({'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 7])}, False),
        ({'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, {'a': np.array([1, 2, 3])}, False),
        ({'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}, {'a': np.array([1, 2, 3]), 'c': np.array([4, 5, 6])}, False),
    ]
)
def test_eq_dict_of_ndarray(d1, d2, expected):
    assert math_utils.eq_dict_of_ndarray(d1, d2) == expected