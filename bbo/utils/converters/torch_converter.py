from typing import Dict, Sequence, List

import torch
from torch import Tensor

from bbo.utils.converters.converter import (
    GroupedFeatureTrialConverter,
    ArrayTrialConverter
)
from bbo.utils.trial import ParameterDict, MetricDict, Trial


class TorchGroupedFeatureTrialConverter(GroupedFeatureTrialConverter):
    def to_features(self, trials: Sequence[Trial]) -> Dict[str, Tensor]:
        grouped_feature = super().to_features(trials)
        for key in grouped_feature:
            grouped_feature[key] = torch.from_numpy(grouped_feature[key])
        return grouped_feature
    
    def to_labels(self, trials: Sequence[Trial]) -> Dict[str, Tensor]:
        labels = super().to_labels(trials)
        for obj_name in labels:
            labels[obj_name] = torch.from_numpy(labels[obj_name])
        return labels
    
    def to_parameters(self, features: Dict[str, Tensor]) -> List[ParameterDict]:
        for name in features:
            features[name] = features[name].detach().to('cpu').numpy()
        return super().to_parameters(features)

    def to_metrics(self, labels: Dict[str, Tensor]) -> List[MetricDict]:
        for obj_name in labels:
            labels[obj_name] = labels[obj_name].detach().to('cpu').numpy()
        return super().to_metrics(labels)


class TorchArrayTrialConverter(ArrayTrialConverter):
    def to_features(self, trials: Sequence[Trial]) -> Tensor:
        features = super().to_features(trials)
        return torch.from_numpy(features)
    
    def to_labels(self, trials: Sequence[Trial]) -> Tensor:
        labels = super().to_labels(trials)
        return torch.from_numpy(labels)
    
    def to_parameters(self, features: Tensor) -> List[ParameterDict]:
        features = features.detach().to('cpu').numpy()
        return super().to_parameters(features)
    
    def to_metrics(self, labels: Tensor) -> List[MetricDict]:
        labels = labels.detach().to('cpu').numpy()
        return super().to_metrics(labels)
