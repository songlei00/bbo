import functools
from typing import Sequence, Callable, Dict

import numpy as np


def arg_operator(data: Sequence, operator: Callable[[Sequence], int]):
    index, _ = operator(enumerate(data), key=lambda x: x[1])
    return index

argmin = functools.partial(arg_operator, operator=min)
argmax = functools.partial(arg_operator, operator=max)


def eq_dict_of_ndarray(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]):
    if len(d1) != len(d2):
        return False
    for key in d1:
        if key not in d2 or not np.allclose(d1[key], d2[key]):
            return False
    return True