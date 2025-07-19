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

import abc
import enum
from typing import Union, Tuple, Optional, Callable, Sequence, List, Dict, Collection
from collections import UserDict

from attrs import define, field, validators, evolve
import numpy as np

from bbo.shared.parameter_config import (
    ParameterConfig,
    ParameterType,
    ParameterValue,
    ScaleType
)
from bbo.shared.trial import Trial, Measurement, Metric, ParameterDict
from bbo.shared.base_study_config import MetricInformation
from bbo.utils import attrs_utils
from bbo.shared.base_study_config import ProblemStatement


class NumpyArraySpecType(enum.Enum):
    DOUBLE = 'DOUBLE'
    INTEGER = 'INTEGER'
    CATEGORICAL = 'CATEGORICAL'
    DISCRETE = 'DISCRETE'
    CUSTOM = 'CUSTOM'
    ONEHOT_EMBEDDING = 'ONEHOT_EMBEDDING'

    @classmethod
    def default_factory(cls, pc: ParameterConfig) -> 'NumpyArraySpecType':
        map_dict = {
            ParameterType.DOUBLE: NumpyArraySpecType.DOUBLE,
            ParameterType.INTEGER: NumpyArraySpecType.INTEGER,
            ParameterType.CATEGORICAL: NumpyArraySpecType.CATEGORICAL,
            ParameterType.DISCRETE: NumpyArraySpecType.DISCRETE,
            ParameterType.CUSTOM: NumpyArraySpecType.CUSTOM,
        }
        return map_dict[pc.type]
    

@define
class NumpyArraySpec:
    name: str = field(validator=validators.instance_of(str), kw_only=True)
    type: NumpyArraySpecType = field(validator=validators.instance_of(NumpyArraySpecType), kw_only=True)
    dtype: np.dtype = field(
        converter=np.dtype,
        validator=validators.in_([np.int32, np.int64, np.float32, np.float64, np.object_]),
        kw_only=True
    )
    bounds: Union[Tuple[float, float], Tuple[int, int]] = field(
        converter=tuple,
        validator=attrs_utils.assert_bounds,
        kw_only=True
    )
    num_dimensions: int = field(
        validator=validators.instance_of(int),
        kw_only=True
    )
    num_oovs: int = field(
        validator=validators.instance_of(int),
        kw_only=True
    )
    scale: Optional[ScaleType] = field(
        default=None,
        validator=validators.optional(validators.instance_of(ScaleType)),
        kw_only=True
    )

    @classmethod
    def from_parameter_config(
        self,
        pc: ParameterConfig,
        type_factory: Callable[[ParameterConfig], NumpyArraySpecType] = NumpyArraySpecType.default_factory,
        float_dtype: np.dtype | str = np.float32,
        int_dtype: np.dtype | str = np.int32,
        *,
        pad_oovs: bool = True
    ):
        type = type_factory(pc)
        if type == NumpyArraySpecType.DOUBLE:
            return NumpyArraySpec(
                name=pc.name,
                type=type,
                dtype=float_dtype,
                bounds=pc.bounds,
                num_dimensions=1,
                num_oovs=0,
                scale=pc.scale_type
            )
        elif type in (
            NumpyArraySpecType.INTEGER,
            NumpyArraySpecType.CATEGORICAL,
            NumpyArraySpecType.DISCRETE
        ):
            if pad_oovs:
                bounds = (0, len(pc.feasible_values))
                num_oovs = 1
            else:
                bounds = (0, len(pc.feasible_values)-1)
                num_oovs = 0
            return NumpyArraySpec(
                name=pc.name,
                type=type,
                dtype=int_dtype,
                bounds=bounds,
                num_dimensions=1,
                num_oovs=num_oovs
            )
        elif type == NumpyArraySpecType.CUSTOM:
            return NumpyArraySpec(
                name=pc.name,
                type=type,
                dtype=np.object_,
                bounds=(0, 0),
                num_dimensions=0,
                num_oovs=0
            )
        else:
            raise ValueError(f'Unknown NumpyArraySpecType: {type}')


class DictOf2DArray(UserDict[str, np.ndarray]):
    def __init__(self, *args, **kwargs):
        self._dim0 = None
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, item: Union[np.ndarray, List]):
        if not isinstance(item, np.ndarray):
            item = np.array(item)
        if self._dim0 is None:
            self._dim0 = item.shape[0]
        if item.ndim != 2:
            raise ValueError(f'DictOf2DArray only support 2D array. Given {item.ndim}')
        if item.shape[0] != self._dim0:
            raise ValueError(f'Dim0 is not same. Current dim0: {self._dim0}. Given {item.shape[0]}')
        return super().__setitem__(key, item)

    def __eq__(self, other: 'DictOf2DArray'):
        if len(self) != len(other):
            return False
        for k in self:
            if k not in other or not np.all(self[k] == other[k]):
                return False
        return True

    def to_array(self) -> np.ndarray:
        # TODO: float32 and int32 will lead to float64 result
        return np.concatenate(list(self.values()), axis=1)

    def to_dict(self, array: np.ndarray) -> 'DictOf2DArray':
        d = dict()
        begin = 0
        for k, v in self.data.items():
            end = begin + v.shape[1]
            d[k] = array[:, begin: end]
            begin = end
        return DictOf2DArray(d)
    
    @property
    def dim0(self):
        return self._dim0


class ModelInputConverter(abc.ABC):
    @abc.abstractmethod
    def convert(self, trials: Sequence[Trial]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def to_parameter_values(self, array: np.ndarray) -> List[Optional[ParameterValue]]:
        pass


@define
class ModelInputArrayBijector:
    forward_fn: Callable[[np.ndarray], np.ndarray] = field()
    backward_fn: Callable[[np.ndarray], np.ndarray] = field()
    output_spec: NumpyArraySpec = field(validator=validators.instance_of(NumpyArraySpec))

    @classmethod
    def identity(cls, spec: NumpyArraySpec):
        return cls(lambda x: x, lambda x: x, spec)

    @classmethod
    def scaler_from_spec(cls, spec: NumpyArraySpec) -> 'ModelInputArrayBijector':
        if spec.type != NumpyArraySpecType.DOUBLE:
            return cls.identity(spec)
        low, high = spec.bounds
        if low == high:
            def forward_fn(x):
                return np.where(np.isfinite(x), 0.5, x)
            def backward_fn(x):
                return np.where(np.isfinite(x), low, x)
            return cls(forward_fn, backward_fn, evolve(spec, bounds=(0.5, 0.5), scale=None))
        else:
            bounds = (0.0, 1.0)
            if spec.scale == ScaleType.LINEAR or spec.scale is None:
                def forward_fn(x):
                    return (x - low) / (high - low)
                def backward_fn(x):
                    return x * (high - low) + low
            elif spec.scale == ScaleType.LOG:
                if low <= 0 or high <= 0:
                    raise ValueError(f'Log scale must be positive. Given {spec.bounds}')
                low, high = np.log(low), np.log(high)
                def forward_fn(x):
                    return (np.log(x) - low) / (high - low)
                def backward_fn(x):
                    return np.exp(x * (high - low) + low)
            elif spec.scale == ScaleType.REVERSE_LOG:
                if low <= 0 or high <= 0:
                    raise ValueError(f'Reverse log scale must be positive. Given {spec.bounds}')
                raw_sum = low + high
                low, high = np.log(low), np.log(high)
                def forward_fn(x):
                    return 1.0 - (np.log(raw_sum - x) - low) / (high - low)
                def backward_fn(x):
                    return raw_sum - np.exp(x * (high - low) + low)
            else:
                raise NotImplementedError(f'Unsupported scale type: {spec.scale}')
            return cls(forward_fn, backward_fn, evolve(spec, bounds=bounds, scale=None))

    @classmethod
    def onehot_embedder_from_spec(cls, spec: NumpyArraySpec, *, dtype: np.float32) -> 'ModelInputArrayBijector':
        if spec.type not in (
            NumpyArraySpecType.DISCRETE,
            NumpyArraySpecType.CATEGORICAL
        ):
            return cls.identity(spec)

        output_spec = NumpyArraySpec(
            name=spec.name,
            type=NumpyArraySpecType.ONEHOT_EMBEDDING,
            dtype=dtype,
            bounds=(0.0, 1.0),
            num_dimensions=(spec.bounds[1] - spec.bounds[0] + 1),
            num_oovs=spec.num_oovs,
            scale=None
        )

        def forward_fn(x):
            return np.eye(output_spec.num_dimensions, dtype=output_spec.dtype)[x.flatten()]
        def backward_fn(x):
            return np.argmax(
                x[:, : output_spec.num_dimensions - output_spec.num_oovs], axis=1
            ).astype(spec.dtype)
        return cls(forward_fn, backward_fn, output_spec)
    

def _create_default_getter(pc: ParameterConfig):
    def getter(trial: Trial):
        pv = trial.parameters[pc.name]
        if pc.type == ParameterType.DOUBLE:
            return pv.as_float()
        elif pc.type == ParameterType.INTEGER:
            return pv.as_int()
        elif pc.type == ParameterType.DISCRETE:
            return pv.as_float()
        elif pc.type == ParameterType.CATEGORICAL:
            return pv.as_str()
        else:
            return pv.value
    return getter


class DefaultModelInputConverter(ModelInputConverter):
    def __init__(
        self,
        parameter_config: ParameterConfig,
        float_dtype: np.dtypes = np.float32,
        int_dtype: np.dtypes = np.int32,
        scale: bool = True,
        onehot_embed: bool = False,
        pad_oovs: bool = False,
        should_clip: bool = True,
        max_discrete_indices: Optional[int] = None
    ):
        self.parameter_config = parameter_config
        if max_discrete_indices is not None and \
            parameter_config.num_feasible_values > max_discrete_indices and \
                parameter_config.type in (ParameterType.INTEGER, ParameterType.DISCRETE):
            parameter_config = parameter_config.continuify()
            self.continuify = True
        else:
            self.continuify = False
        self.spec = NumpyArraySpec.from_parameter_config(
            parameter_config,
            NumpyArraySpecType.default_factory,
            float_dtype,
            int_dtype,
            pad_oovs=pad_oovs
        )
        self.scaler = (
            ModelInputArrayBijector.identity(self.spec)
            if not scale else
            ModelInputArrayBijector.scaler_from_spec(self.spec)
        )
        self.onehot_embedder = (
            ModelInputArrayBijector.identity(self.scaler.output_spec)
            if not onehot_embed else
            ModelInputArrayBijector.onehot_embedder_from_spec(self.scaler.output_spec, dtype=float_dtype)
        )
        self.output_spec = self.onehot_embedder.output_spec
        self.should_clip = should_clip
        self.getter = _create_default_getter(parameter_config)

    def convert(self, trials: Sequence[Trial]) -> np.ndarray:
        if not trials:
            return np.zeros((0, self.output_spec.num_dimensions), dtype=self.output_spec.dtype)
        convert_fn = (
            self._convert_index if self.spec.type in (
                NumpyArraySpecType.INTEGER, NumpyArraySpecType.DISCRETE, NumpyArraySpecType.CATEGORICAL
            ) else
            self._convert_continuous
        )
        values = [convert_fn(trial) for trial in trials]
        array = np.asarray(values, dtype=self.spec.dtype).reshape(-1, 1)
        
        return self.onehot_embedder.forward_fn(self.scaler.forward_fn(array))
            
    def _convert_index(self, trial: Trial):
        raw_value = self.getter(trial)
        if raw_value in self.parameter_config.feasible_values:
            return self.parameter_config.feasible_values.index(raw_value)
        else:
            if self.spec.num_oovs > 0:
                return len(self.parameter_config.feasible_values)
            else:
                raise ValueError(f'Value {raw_value} is not in feasible values {self.parameter_config.feasible_values} for parameter {self.parameter_config.name}')

    def _convert_continuous(self, trial: Trial):
        raw_value = self.getter(trial)
        return raw_value

    def to_parameter_values(self, array: np.ndarray) -> List[Optional[ParameterValue]]:
        array = self.onehot_embedder.backward_fn(self.scaler.backward_fn(array))
        return [self._to_parameter_value(v) for v in array.flatten()]
    
    def _to_parameter_value(self, value):
        if self.parameter_config.type == ParameterType.DOUBLE:
            if self.should_clip:
                value = np.clip(value, self.parameter_config.bounds[0], self.parameter_config.bounds[1])
            return ParameterValue(float(value))
        elif self.parameter_config.type in (
            ParameterType.INTEGER, ParameterType.DISCRETE, ParameterType.CATEGORICAL
        ):
            if self.continuify:
                diffs = np.abs(np.asarray(self.parameter_config.feasible_values)- value)
                idx = np.argmin(diffs)
            else:
                idx = value
            return ParameterValue(self.parameter_config.feasible_values[int(idx)])
        else:
            return ParameterValue(value)
        

class ModelOutputConverter(abc.ABC):
    @abc.abstractmethod
    def convert(self, measurements: Sequence[Measurement]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def to_metrics(self, labels: np.ndarray) -> List[Optional[Metric]]:
        pass


class DefaultModelOutputConverter(ModelOutputConverter):
    def __init__(
        self,
        metric_information: MetricInformation,
        dtype: np.dtypes = np.float32
    ):
        self.metric_information = metric_information
        self.dtype = dtype

    def convert(self, measurements: Sequence[Measurement]) -> np.ndarray:
        if not measurements:
            return np.zeros((0, 1), dtype=self.dtype)
        labels = [m.metrics[self.metric_information.name].value for m in measurements]
        labels = np.asarray(labels, dtype=self.dtype)
        return labels.reshape(-1, 1)

    def to_metrics(self, array: np.ndarray) -> List[Optional[Metric]]:
        return [Metric(v) for v in array.flatten()]


class TrialConverter(abc.ABC):
    @abc.abstractmethod
    def to_features(self, trials: Sequence[Trial]):
        pass

    @abc.abstractmethod
    def to_labels(self, trials: Sequence[Trial]):
        pass

    @abc.abstractmethod
    def to_xy(self, trials: Sequence[Trial]):
        pass

    @abc.abstractmethod
    def to_parameters(self, features: Dict[str, np.ndarray]):
        pass

    @abc.abstractmethod
    def to_measurements(self, labels: Dict[str, np.ndarray]):
        pass

    @abc.abstractmethod
    def to_trials(self, features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]):
        pass

    @property
    def parameter_configs(self) -> Dict[str, ParameterConfig]:
        pass

    @property
    def output_specs(self) -> Dict[str, NumpyArraySpec]:
        pass

    @property
    def metric_information(self) -> Dict[str, MetricInformation]:
        pass


class DefaultTrialConverter(TrialConverter):
    def __init__(
        self,
        parameter_converters: Collection[ModelInputConverter],
        output_converters: Collection[ModelOutputConverter]
    ):
        self.parameter_converters = {
            converter.parameter_config.name: converter
            for converter in parameter_converters
        }
        self.output_converters = {
            converter.metric_information.name: converter
            for converter in output_converters
        }

    def to_features(self, trials: Sequence[Trial]) -> Dict[str, np.ndarray]:
        d = dict()
        for name, converter in self.parameter_converters.items():
            d[name] = converter.convert(trials)
        return d

    def to_labels(self, trials: Sequence[Trial]) -> Dict[str, np.ndarray]:
        d = dict()
        for name, converter in self.output_converters.items():
            d[name] = converter.convert([t.final_measurement for t in trials])
        return d

    def to_xy(self, trials: Sequence[Trial]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        return self.to_features(trials), self.to_labels(trials)

    def to_parameters(self, features: Dict[str, np.ndarray]) -> List[ParameterDict]:
        features = DictOf2DArray(features)
        parameters = [ParameterDict() for _ in range(features.dim0)]
        for name, array in features.items():
            converter = self.parameter_converters[name]
            parameter_values = converter.to_parameter_values(array)
            for p, v in zip(parameters, parameter_values):
                p[name] = v
        return parameters

    def to_measurements(self, labels: Dict[str, np.ndarray]) -> List[Measurement]:
        labels = DictOf2DArray(labels)
        measurements = [Measurement() for _ in range(labels.dim0)]
        for name, array in labels.items():
            converter = self.output_converters[name]
            metrics = converter.to_metrics(array)
            for metric, measurement in zip(metrics, measurements):
                measurement.metrics[name] = metric
        return measurements

    def to_trials(self, features: Dict[str, np.ndarray], labels: Dict[str, np.ndarray]) -> List[Trial]:
        parameters = self.to_parameters(features)
        measurements = self.to_measurements(labels)
        return self._agg2trials(parameters, measurements)

    def _agg2trials(self, parameters: List[ParameterDict], measurements: List[Measurement]) -> List[Trial]:
        trials = []
        for p, m in zip(parameters, measurements):
            t = Trial(p, final_measurement=m)
            trials.append(t)
        return trials
    
    @classmethod
    def from_study_config(
        cls,
        study_config: ProblemStatement,
        float_dtype: np.dtypes = np.float32,
        int_dtype: np.dtypes = np.int32,
        scale: bool = True,
        onehot_embed: bool = False,
        pad_oovs: bool = False,
        should_clip: bool = True,
        max_discrete_indices: int | None = None
    ):
        parameter_converters = [
            DefaultModelInputConverter(p, float_dtype, int_dtype, scale, onehot_embed, pad_oovs, should_clip, max_discrete_indices)
            for p in study_config.search_space.parameter_configs.values()
        ]
        output_converters = [
            DefaultModelOutputConverter(m, float_dtype)
            for m in study_config.metric_information
        ]
        return cls(parameter_converters, output_converters)
    
    @property
    def parameter_configs(self) -> Dict[str, ParameterConfig]:
        return {name: converter.parameter_config for name, converter in self.parameter_converters.items()}

    @property
    def output_specs(self) -> Dict[str, NumpyArraySpec]:
        return {name: converter.output_spec for name, converter in self.parameter_converters.items()}

    @property
    def metric_information(self) -> Dict[str, MetricInformation]:
        return {name: converter.metric_information for name, converter in self.output_converters.items()}


@define
class TrialToArrayConverter(TrialConverter):
    _impl: TrialConverter = field(validator=validators.instance_of(TrialConverter))

    def to_features(self, trials: Sequence[Trial]) -> np.ndarray:
        d = DictOf2DArray(self._impl.to_features(trials))
        return d.to_array()

    def to_labels(self, trials: Sequence[Trial]) -> np.ndarray:
        d = DictOf2DArray(self._impl.to_labels(trials))
        return d.to_array()

    def to_xy(self, trials: Sequence[Trial]) -> Tuple[np.ndarray, np.ndarray]:
        return self.to_features(trials), self.to_labels(trials)

    def to_parameters(self, features: np.ndarray) -> List[ParameterDict]:
        d = DictOf2DArray(self._impl.to_features([]))
        return self._impl.to_parameters(d.to_dict(features))

    def to_measurements(self, labels: np.ndarray) -> List[Measurement]:
        d = DictOf2DArray(self._impl.to_labels([]))
        return self._impl.to_measurements(d.to_dict(labels))

    def to_trials(self, features: np.ndarray, labels: np.ndarray) -> List[Trial]:
        parameters = self.to_parameters(features)
        measurements = self.to_measurements(labels)
        return self._impl._agg2trials(parameters, measurements)

    @classmethod
    def from_study_config(
        cls,
        study_config: ProblemStatement,
        float_dtype: np.dtypes = np.float32,
        int_dtype: np.dtypes = np.int32,
        scale: bool = True,
        onehot_embed: bool = False,
        pad_oovs: bool = False,
        should_clip: bool = True,
        max_discrete_indices: int | None = None
    ):

        return cls(DefaultTrialConverter.from_study_config(
            study_config, float_dtype, int_dtype, scale, onehot_embed, pad_oovs, should_clip, max_discrete_indices
        ))

    @property
    def parameter_configs(self) -> Dict[str, ParameterConfig]:
        return self._impl.parameter_configs

    @property
    def output_specs(self) -> Dict[str, NumpyArraySpec]:
        return self._impl.output_specs

    @property
    def metric_information(self) -> Dict[str, MetricInformation]:
        return self._impl.metric_information