import abc

import torch
from attrs import define, field, validators


class AcquisitionFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, mu: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
        pass


@define
class EI(AcquisitionFunction):
    _best_f: float = field(converter=float, validator=validators.instance_of(float))
    _eps: float = field(default=1e-6, converter=float, validator=validators.instance_of(float))

    def __call__(self, mu: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
        improvement = mu - self._best_f - self._eps
        Z = improvement / stddev
        ei = improvement * torch.distributions.Normal(0.0, 1.0).cdf(Z) + \
            stddev * torch.distributions.Normal(0.0, 1.0).log_prob(Z).exp()
        return ei
    

@define
class UCB(AcquisitionFunction):
    beta: float = field(default=1.8, converter=float, validator=validators.instance_of(float))

    def __call__(self, mu: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
        return mu + self.beta * stddev