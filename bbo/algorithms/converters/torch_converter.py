# Copyright 2025 songlei
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Sequence, List, Optional

import numpy as np
import torch
from attrs import define, field, validators

from bbo.algorithms.converters.core import TrialConverter, _agg2trials, TypeArray, TrialToTypeArrayConverter
from bbo.shared.trial import Trial, ParameterDict, Measurement
from bbo.shared.base_study_config import ProblemStatement


@define
class TrialToTorchConverter(TrialConverter):
    _impl: TrialConverter = field(validator=validators.instance_of(TrialConverter))

    def to_features(self, trials: Sequence[Trial]) -> TypeArray:
        np_features = self._impl.to_features(trials)
        return TypeArray(
            double=torch.from_numpy(np_features.double),
            integer=torch.from_numpy(np_features.integer),
            categorical=torch.from_numpy(np_features.categorical),
            discrete=torch.from_numpy(np_features.discrete)
        )

    def to_labels(self, trials: Sequence[Trial]) -> torch.Tensor:
        return torch.from_numpy(self._impl.to_labels(trials))

    def to_xy(self, trials: Sequence[Trial]) -> Tuple[TypeArray, torch.Tensor]:
        return self.to_features(trials), self.to_labels(trials)

    def to_parameters(self, features: TypeArray) -> List[ParameterDict]:
        np_features = TypeArray(
            double=features.double.numpy(),
            integer=features.integer.numpy(),
            categorical=features.categorical.numpy(),
            discrete=features.discrete.numpy()
        )
        return self._impl.to_parameters(np_features)
    
    def to_measurements(self, labels: torch.Tensor) -> List[Measurement]:
        return self._impl.to_measurements(labels.numpy())

    def to_trials(self, features: TypeArray, labels: Optional[torch.Tensor] = None) -> List[Trial]:
        parameters = self.to_parameters(features)
        measurements = self.to_measurements(labels) if labels is not None else None
        return _agg2trials(parameters, measurements)
    
    @classmethod
    def from_study_config(
        cls,
        study_config: ProblemStatement,
        float_dtype: np.dtypes = np.float32,
        int_dtype: np.dtypes = np.int32,
        scale: bool = True,
        pad_oovs: bool = False,
        should_clip: bool = True,
        max_discrete_indices: int | None = None
    ):
        return cls(
            TrialToTypeArrayConverter.from_study_config(
                study_config, float_dtype, int_dtype, scale, pad_oovs, should_clip, max_discrete_indices
            )
        )