from typing import Dict, Sequence, List, Tuple

import torch
from torch import Tensor

from bbo.utils.converters.base import (
    BaseInputConverter,
    BaseOutputConverter,
    BaseTrialConverter,
)
from bbo.utils.converters.converter import ArrayTrialConverter, NumpyArraySpec
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.metric_config import MetricInformation
from bbo.utils.trial import ParameterDict, MetricDict, Trial


class TorchTrialConverter(BaseTrialConverter):
    def __init__(
        self,
        input_converters: Sequence[BaseInputConverter],
        output_converters: Sequence[BaseOutputConverter],
    ):
        self._impl = ArrayTrialConverter(input_converters, output_converters)

    @classmethod
    def from_problem(
        cls,
        problem: ProblemStatement,
        *,
        scale: bool = True,
        onehot_embed: bool = False,
    ):
        converter = cls([], [])
        converter._impl = ArrayTrialConverter.from_problem(
            problem, scale=scale, onehot_embed=onehot_embed
        )
        return converter

    def convert(self, trials: Sequence[Trial]) -> Tuple[Tensor, Tensor]:
        return self.to_features(trials), self.to_labels(trials)

    def to_features(self, trials: Sequence[Trial]) -> Tensor:
        return torch.from_numpy(self._impl.to_features(trials))

    def to_labels(self, trials: Sequence[Trial]) -> Tensor:
        return torch.from_numpy(self._impl.to_labels(trials))

    def to_trials(self, features: Tensor, labels: Tensor=None) -> Sequence[Trial]:
        features = features.detach().numpy()
        if labels is not None:
            labels = labels.detach().numpy()
        return self._impl.to_trials(features, labels)

    def to_parameters(self, features: Tensor) -> List[ParameterDict]:
        return self._impl.to_parameters(features.detach().numpy())

    def to_metrics(self, labels: Tensor) -> List[MetricDict]:
        return self._impl.to_metrics(labels.detach().numpy())

    @property
    def output_spec(self) -> Dict[str, NumpyArraySpec]:
        return {k: v.output_spec for k, v in self._impl.input_converter_dict.items()}

    @property
    def metric_spec(self) -> Dict[str, MetricInformation]:
        return {k: v.metric_information for k, v in self._impl.output_converter_dict.items()}