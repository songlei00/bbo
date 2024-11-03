from abc import ABC, abstractmethod
from typing import List
from functools import partial

from attrs import define, field, validators
import numpy as np
import torch
import torch.nn as nn


class Embedder(ABC):
    @abstractmethod
    def map(self, data):
        pass

    @abstractmethod
    def unmap(self, data):
        pass


@define
class OneHotEmbedder(Embedder):
    _num_feasible_values: List = field(validator=validators.instance_of(list))
    
    def __attrs_post_init__(self):
        self._dim = len(self._num_feasible_values)

    def map(self, data):
        dim = data.shape[-1]
        assert dim == self._dim
        if isinstance(data, np.ndarray):
            eye_fn = np.eye
            cat_fn = partial(np.concatenate, axis=-1)
        elif isinstance(data, torch.Tensor):
            eye_fn = torch.eye
            cat_fn = partial(torch.cat, dim=-1)
        else:
            raise TypeError('Only support np.ndarray and torch.Tensor')
        
        onehot_data = []
        for i, n in enumerate(self._num_feasible_values):
            onehot_data.append(eye_fn(n)[data[:, i]])
        return cat_fn(onehot_data)

    def unmap(self, onehot_data):
        dim = onehot_data.shape[-1]
        assert dim == sum(self._num_feasible_values)
        if isinstance(onehot_data, np.ndarray):
            argmax_fn = partial(np.argmax, axis=-1, keepdims=True)
            cat_fn = partial(np.concatenate, axis=-1)
        elif isinstance(onehot_data, torch.Tensor):
            argmax_fn = partial(torch.argmax, axis=-1, keepdim=True)
            cat_fn = partial(torch.cat, dim=-1)
        else:
            raise TypeError('Only support np.ndarray and torch.Tensor')
        start_idx = 0
        data = []
        for n in self._num_feasible_values:
            data.append(argmax_fn(onehot_data[:, start_idx: start_idx+n]))
            start_idx += n
        return cat_fn(data)


@define
class DenseEmbedder(Embedder):
    _num_feasible_values: List = field(validator=validators.instance_of(list))
    _d: int = field(validator=validators.instance_of(int))
    _contextual_embedding: bool = field(default=False, kw_only=True)
    _contextual_config: dict = field(factory=dict, kw_only=True)

    def __attrs_post_init__(self):
        self.embed_layer = nn.Embedding(sum(self._num_feasible_values), self._d)
        self.start_idx_tensor = torch.tensor(
            [sum(self._num_feasible_values[:i]) for i in range(len(self._num_feasible_values))]
        )

    def map(self, data):
        data += self.start_idx_tensor
        dense_data = self.embed_layer(data)
        if self._contextual_embedding:
            # TODO: contextual embedding
            raise NotImplementedError('Contextual embedding')
        dense_data = torch.flatten(dense_data, start_dim=-2, end_dim=-1)
        return dense_data        

    def unmap(self, data):
        raise RuntimeError('Can not convert dense embdding to index')
