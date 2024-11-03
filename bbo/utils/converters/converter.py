import enum
import logging
from typing import Union, Dict, List, Tuple, Callable, Optional, Sequence
from collections import defaultdict

import numpy as np
from attrs import define, field, validators, evolve

from bbo.utils.converters.base import (
    BaseInputConverter,
    BaseOutputConverter,
    BaseTrialConverter,
)
from bbo.utils.parameter_config import (
    ParameterType,
    ParameterConfig,
    ScaleType,
)
from bbo.utils.metric_config import MetricInformation
from bbo.utils.trial import (
    ParameterValue,
    ParameterDict,
    Metric,
    MetricDict,
    Trial,
)
from bbo.utils.problem_statement import ProblemStatement

logger = logging.getLogger(__name__)


class SpecType(enum.Enum):
    """Type information from the algorithmic perspective"""
    DOUBLE = 'DOUBLE'
    INTEGER = 'INTEGER'
    DISCRETE = 'DISCRETE'
    CATEGORICAL = 'CATEGORICAL'


@define(frozen=True)
class FeatureSpec:
    """Encode the data that can be used as training data directly

    This class is a countpart of ParameterConfig class
    """
    name: str = field(
        validator=validators.instance_of(str),
    )
    type: SpecType = field(
        validator=validators.instance_of(SpecType),
    )
    dtype: np.dtype = field(
        validator=validators.in_([np.int32, np.int64, np.float32, np.float64,])
    )
    bounds: Union[Tuple[float, float], Tuple[int, int]] = field(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of((float, int)),
            iterable_validator=validators.instance_of(tuple)
        ),
    )
    scale_type: Optional[ScaleType] = field(
        default=None,
        validator=validators.optional(
            validators.instance_of(ScaleType)
        ),
    )
    num_oovs: int = field(default=0, validator=validators.instance_of(int))

    @classmethod
    def from_parameter_config(
        cls,
        pc: ParameterConfig,
        *,
        float_dtype: np.dtype = np.float32,
        int_dtype: np.dtype = np.int32,
        num_oovs: int = 0
    ):
        spec_type = {
            ParameterType.DOUBLE: SpecType.DOUBLE,
            ParameterType.INTEGER: SpecType.INTEGER,
            ParameterType.DISCRETE: SpecType.DISCRETE,
            ParameterType.CATEGORICAL: SpecType.CATEGORICAL
        }[pc.type]
        if spec_type == SpecType.DOUBLE:
            return FeatureSpec(
                name=pc.name,
                type=spec_type,
                dtype=float_dtype,
                bounds=pc.bounds,
                scale_type=pc.scale_type,
                num_oovs=num_oovs
            )
        elif spec_type in (
            SpecType.INTEGER,
            SpecType.CATEGORICAL,
            SpecType.DISCRETE,
        ):
            return FeatureSpec(
                name=pc.name,
                type=spec_type,
                dtype=int_dtype,
                bounds=(0, len(pc.feasible_values)-1+num_oovs),
                scale_type=pc.scale_type,
                num_oovs=num_oovs
            )
        else:
            raise ValueError('Unknown type: {}'.format(spec_type))


@define
class FeatureSpecTransform:
    forward_fn: Callable[[np.ndarray], np.ndarray]
    backward_fn: Callable[[np.ndarray], np.ndarray]
    output_spec: FeatureSpec

    @classmethod
    def identity(cls, spec: FeatureSpec):
        return cls(lambda x: x, lambda x: x, spec)

    @classmethod
    def scaler(cls, spec: FeatureSpec):
        if spec.type != SpecType.DOUBLE:
            return cls.identity(spec)
        
        if spec.scale_type is None:
            return cls.identity(spec)
        elif spec.scale_type == ScaleType.LINEAR:
            lb, ub = spec.bounds
            def forward_fn(x, lb=lb, ub=ub):
                return (x - lb) / (ub - lb)
            def backward_fn(x, lb=lb, ub=ub):
                return x * (ub - lb) + lb
            return cls(
                forward_fn,
                backward_fn,
                evolve(spec, bounds=(0.0, 1.0), scale_type=None)
            )
        elif spec.scale_type == ScaleType.LOG:
            lb, ub = spec.bounds
            lb, ub = np.log(lb), np.log(ub)
            def forward_fn(x, lb=lb, ub=ub):
                return (np.log(x) - lb) / (ub - lb)
            def backward_fn(x, lb=lb, ub=ub):
                return np.exp(x * (ub - lb) + lb)
            return cls(
                forward_fn,
                backward_fn,
                evolve(spec, bounds=(0.0, 1.0), scale_type=None),
            )
        else:
            raise NotImplementedError('ScaleType {} is not supported'.format(spec.scale_type))
        

class DefaultInputConverter(BaseInputConverter):
    def __init__(
        self,
        pc: ParameterConfig,
        *,
        scale: bool = True,
        num_oovs: int = 0
    ):
        self._pc = pc
        self.spec = FeatureSpec.from_parameter_config(pc, num_oovs=num_oovs)

        spec = self.spec
        self.scaler = (
            FeatureSpecTransform.scaler(spec)
            if scale
            else FeatureSpecTransform.identity(spec)
        )
        self._output_spec = self.scaler.output_spec

    def convert(self, trials: Sequence[Trial]) -> np.ndarray:
        if not trials:
            return np.zeros((0, 1), dtype=self._output_spec.dtype)

        if self.spec.type == SpecType.DOUBLE:
            values = [t.parameters[self._pc.name].value for t in trials]
        elif self.spec.type in (
            SpecType.INTEGER,
            SpecType.CATEGORICAL,
            SpecType.DISCRETE,
        ):
            values = [
                self._pc.feasible_values.index(t.parameters[self._pc.name].value)
                for t in trials
            ]
        else:
            raise NotImplementedError('type {} is not supported'.format(self._pc.type))
        values = np.array(values).reshape(-1, 1)
        
        return self.scaler.forward_fn(values)
    
    def to_parameter_values(self, array: np.ndarray) -> List[ParameterValue]:
        values = self.scaler.backward_fn(array)
        values = values.flatten()
        if self.spec.type == SpecType.DOUBLE:
            return [ParameterValue(v) for v in values]
        elif self.spec.type in (
            SpecType.INTEGER,
            SpecType.CATEGORICAL,
            SpecType.DISCRETE,
        ):
            if not np.all(np.isclose(values, values.astype(int))):
                logger.warning('Float values are round to integer')
            return [ParameterValue(self._pc.feasible_values[int(v)]) for v in values]

    @property
    def output_spec(self):
        return self._output_spec

    @property
    def parameter_config(self):
        return self._pc


class DefaultOutputConverter(BaseOutputConverter):
    def __init__(
        self,
        metric_information: MetricInformation,
        # *,
        # normalize: bool = False, # TODO: add normalize
    ):
        self._metric_information = metric_information

    def convert(self, trials: Sequence[Trial]) -> np.ndarray:
        labels = [t.metrics[self._metric_information.name].value for t in trials]
        labels = np.array(labels).reshape(-1, 1)
        return labels

    def to_metrics(self, array: np.ndarray) -> List[Metric]:
        metrics = [Metric(v) for v in array.flatten()]
        return metrics

    @property
    def metric_information(self) -> MetricInformation:
        return self._metric_information


class DefaultTrialConverter(BaseTrialConverter):
    def __init__(
        self,
        input_converters: Sequence[BaseInputConverter],
        output_converters: Sequence[BaseOutputConverter]
    ):
        self.input_converter_dict = {
            converter.parameter_config.name: converter for converter in input_converters
        }
        self.output_converter_dict = {
            converter.metric_information.name: converter for converter in output_converters
        }

    @classmethod
    def from_problem(
        cls,
        problem: ProblemStatement,
        *,
        scale: bool = True,
        num_oovs: int = 0
    ):
        parameter_configs = problem.search_space.parameters
        objectives = problem.objective.metrics

        input_converters = [
            DefaultInputConverter(pc, scale=scale, num_oovs=num_oovs)
            for pc in parameter_configs
        ]
        output_converters = [DefaultOutputConverter(m) for m in objectives]

        return cls(input_converters, output_converters)

    def to_features(self, trials: Sequence[Trial]) -> Dict[str, np.ndarray]:
        features = dict()
        for name, converter in self.input_converter_dict.items():
            features[name] = converter.convert(trials)
        return features

    def to_labels(self, trials: Sequence[Trial]) -> Dict[str, np.ndarray]:
        labels = dict()
        for name, convert in self.output_converter_dict.items():
            labels[name] = convert.convert(trials)
        return labels

    def to_parameters(self, features: Dict[str, np.ndarray]) -> List[ParameterDict]:
        size = next(iter(features.values())).shape[0]
        parameters = [ParameterDict() for _ in range(size)]
        for name, converter in self.input_converter_dict.items():
            parameter_values = converter.to_parameter_values(features[name])
            for p, v in zip(parameters, parameter_values):
                p[name] = v
        return parameters
        
    def to_metrics(self, labels: Dict[str, np.ndarray]) -> List[MetricDict]:
        size = next(iter(labels.values())).shape[0]
        metrics = [MetricDict() for _ in range(size)]
        for name, converter in self.output_converter_dict.items():
            metric_values = converter.to_metrics(labels[name])
            for m, v in zip(metrics, metric_values):
                m[name] = v
        return metrics

    @property
    def output_spec(self) -> Dict[str, FeatureSpec]:
        return {k: v.output_spec for k, v in self.input_converter_dict.items()}

    @property
    def metric_spec(self) -> Dict[str, MetricInformation]:
        return {k: v.metric_information for k, v in self.output_converter_dict.items()}


class GroupedFeatureTrialConverter(BaseTrialConverter):
    def __init__(
        self,
        input_converters: Sequence[BaseInputConverter],
        output_converters: Sequence[BaseOutputConverter],
    ):
        self._impl = DefaultTrialConverter(input_converters, output_converters)
        self._type2name = dict()
        for spec_type in SpecType:
            self._type2name[spec_type.name] = []
        for name, c in self._impl.input_converter_dict.items():
            spec_type = c.output_spec.type
            self._type2name[spec_type.name].append(name)

    @classmethod
    def from_problem(
        cls,
        problem: ProblemStatement,
        *,
        scale: bool = True,
        num_oovs: int = 0
    ):
        parameter_configs = problem.search_space.parameters
        objectives = problem.objective.metrics

        input_converters = [
            DefaultInputConverter(pc, scale=scale, num_oovs=num_oovs)
            for pc in parameter_configs
        ]
        output_converters = [DefaultOutputConverter(m) for m in objectives]

        return cls(input_converters, output_converters)

    def to_features(self, trials: Sequence[Trial]) -> Dict[str, np.ndarray]:
        features = self._impl.to_features(trials)
        type2array = defaultdict(list)
        for key in self._type2name:
            for name in self._type2name[key]:
                type2array[key].append(features[name])
        for key in self._type2name:
            if len(type2array[key]) == 0:
                type2array[key] = np.zeros((len(trials), 0))
            else:
                type2array[key] = np.concatenate(type2array[key], axis=-1)
        return dict(type2array)

    def to_labels(self, trials: Sequence[Trial]) -> Dict[str, np.ndarray]:
        return self._impl.to_labels(trials)
    
    def to_parameters(self, features: Dict[str, np.ndarray]) -> List[ParameterDict]:
        splitted_features = dict()
        for key in self._type2name:
            for i, name in enumerate(self._type2name[key]):
                splitted_features[name] = features[key][:, i]
        features = splitted_features
        return self._impl.to_parameters(features)
        
    def to_metrics(self, labels: Dict[str, np.ndarray]) -> List[MetricDict]:
        return self._impl.to_metrics(labels)

    @property
    def output_spec(self) -> Dict[str, FeatureSpec]:
        return {k: v.output_spec for k, v in self._impl.input_converter_dict.items()}

    @property
    def metric_spec(self) -> Dict[str, MetricInformation]:
        return {k: v.metric_information for k, v in self._impl.output_converter_dict.items()}
    

def dict2array(d: Dict) -> np.ndarray:
    return np.concatenate(list(d.values()), axis=-1)


class ArrayTrialConverter(BaseTrialConverter):
    def __init__(
        self,
        input_converters: Sequence[BaseInputConverter],
        output_converters: Sequence[BaseOutputConverter],
    ):
        self._impl = DefaultTrialConverter(input_converters, output_converters)

    @classmethod
    def from_problem(
        cls,
        problem: ProblemStatement,
        *,
        scale: bool = True,
        onehot_embed: bool = False,
    ):
        converter = cls([], [])
        converter._impl = DefaultTrialConverter.from_problem(
            problem, scale=scale, onehot_embed=onehot_embed
        )
        return converter

    def to_features(self, trials: Sequence[Trial]) -> np.ndarray:
        return dict2array(self._impl.to_features(trials))

    def to_labels(self, trials: Sequence[Trial]) -> np.ndarray:
        return dict2array(self._impl.to_labels(trials))

    def to_parameters(self, features: np.ndarray) -> List[ParameterDict]:
        size = features.shape[0]
        parameters = [ParameterDict() for _ in range(size)]
        feature_fmt = self._impl.to_features([])
        start, end = 0, 0
        for name, converter in self._impl.input_converter_dict.items():
            end += feature_fmt[name].shape[-1]
            parameter_values = converter.to_parameter_values(features[:, start: end])
            start = end
            for p, v in zip(parameters, parameter_values):
                p[name] = v
        return parameters

    def to_metrics(self, labels: np.ndarray) -> List[MetricDict]:
        size = labels.shape[0]
        metrics = [MetricDict() for _ in range(size)]
        curr = 0
        for name, converter in self._impl.output_converter_dict.items():
            metric_values = converter.to_metrics(labels[:, curr])
            curr += 1
            for m, v in zip(metrics, metric_values):
                m[name] = v
        return metrics

    @property
    def output_spec(self) -> Dict[str, FeatureSpec]:
        return {k: v.output_spec for k, v in self._impl.input_converter_dict.items()}

    @property
    def metric_spec(self) -> Dict[str, MetricInformation]:
        return {k: v.metric_information for k, v in self._impl.output_converter_dict.items()}