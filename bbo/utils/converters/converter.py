import enum
from typing import Union, Dict, List, Tuple, Callable, Optional, Sequence

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


class NumpyArraySpecType(enum.Enum):
    DOUBLE = 'DOUBLE'
    INTEGER = 'INTEGER'
    CATEGORICAL = 'CATEGORICAL'
    DISCRETE = 'DISCRETE'
    ONEHOT_EMBEDDING = 'ONEHOT_EMBEDDING'

    @classmethod
    def default_factory(cls, pc: ParameterConfig):
        map_dict = {
            ParameterType.DOUBLE: NumpyArraySpecType.DOUBLE,
            ParameterType.INTEGER: NumpyArraySpecType.INTEGER,
            ParameterType.CATEGORICAL: NumpyArraySpecType.CATEGORICAL,
            ParameterType.DISCRETE: NumpyArraySpecType.DISCRETE,
        }
        if pc.type in map_dict:
            return map_dict[pc.type]
        else:
            raise ValueError('Unknown type: {}'.format(pc.type))

    @classmethod
    def embedding_factory(cls, pc: ParameterConfig):
        map_dict = {
            ParameterType.DOUBLE: NumpyArraySpecType.DOUBLE,
            ParameterType.INTEGER: NumpyArraySpecType.INTEGER,
            ParameterType.CATEGORICAL: NumpyArraySpecType.ONEHOT_EMBEDDING,
            ParameterType.DISCRETE: NumpyArraySpecType.DISCRETE,
        }

        if pc.type in map_dict:
            return map_dict[pc.type]
        else:
            raise ValueError('Unknown type: {}'.format(pc.type))


@define(frozen=True)
class NumpyArraySpec:
    """Encode the data that can be used as training data directly

    This class is a countpart of ParameterConfig class
    """
    name: str = field(
        validator=validators.instance_of(str),
    )
    type: NumpyArraySpecType = field(
        validator=validators.instance_of(NumpyArraySpecType),
    )
    dtype: np.dtype = field(
        converter=np.dtype,
        validator=validators.in_([np.int32, np.int64, np.float32, np.float64]),
    )
    bounds: Union[Tuple[float, float], Tuple[int, int]] = field(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of((float, int)),
            iterable_validator=validators.instance_of(tuple)
        ),
    )
    num_dimensions: int = field(
        validator=validators.instance_of(int),
    )
    scale_type: Optional[ScaleType] = field(
        default=None,
        validator=validators.optional(
            validators.instance_of(ScaleType)
        ),
    )

    @classmethod
    def from_parameter_config(
        cls,
        pc: ParameterConfig,
        *,
        type_factory: Callable[[ParameterConfig], NumpyArraySpecType] = NumpyArraySpecType.default_factory,
        float_dtype: np.dtype = np.float32,
        int_dtype: np.dtype = np.int32,
    ):
        spec_type = type_factory(pc)
        if spec_type == NumpyArraySpecType.DOUBLE:
            return NumpyArraySpec(
                name=pc.name,
                type=spec_type,
                dtype=float_dtype,
                bounds=pc.bounds,
                num_dimensions=1,
                scale_type=pc.scale_type,
            )
        elif spec_type in (
            NumpyArraySpecType.INTEGER,
            NumpyArraySpecType.CATEGORICAL,
            NumpyArraySpecType.DISCRETE,
        ):
            # We convert INTEGER, CATEGORICAL, DISCRETE parameter configs to 
            # the same NumpyArraySpec and keep the type so that algorithms can 
            # distinguish different parameters and use different optimization
            # strategies
            # 
            # For DISCRETE parameters, the convert will ignore the absolute
            # value but keep the ralative ordering, I don't know if it is a 
            # big deal
            return NumpyArraySpec(
                name=pc.name,
                type=spec_type,
                dtype=int_dtype,
                bounds=(0, len(pc.feasible_values)-1),
                num_dimensions=1,
                scale_type=pc.scale_type,
            )
        elif spec_type == NumpyArraySpecType.ONEHOT_EMBEDDING:
            return NumpyArraySpec(
                name=pc.name,
                type=spec_type,
                dtype=float_dtype,
                bounds=(0.0, 1.0),
                num_dimensions=len(pc.feasible_values),
                scale_type=pc.scale_type,
            )
        else:
            raise ValueError('Unknown type: {}'.format(spec_type))


@define
class SpecTransform:
    forward_fn: Callable[[np.ndarray], np.ndarray]
    backward_fn: Callable[[np.ndarray], np.ndarray]
    output_spec: NumpyArraySpec

    @classmethod
    def identity(cls, spec: NumpyArraySpec):
        return cls(lambda x: x, lambda x: x, spec)

    @classmethod
    def scaler(cls, spec: NumpyArraySpec):
        if spec.type != NumpyArraySpecType.DOUBLE:
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

    @classmethod
    def onehot_embedder(cls, spec: NumpyArraySpec):
        if spec.type != NumpyArraySpecType.CATEGORICAL:
            return cls.identity(spec)

        output_spec = NumpyArraySpec(
            name=spec.name,
            type=NumpyArraySpecType.ONEHOT_EMBEDDING,
            dtype=spec.dtype,
            bounds=(0.0, 1.0),
            num_dimensions=spec.bounds[1]-spec.bounds[0]+1,
            scale_type=None,
        )

        def forward_fn(x):
            return np.eye(output_spec.num_dimensions)[x.flatten()]
        def backward_fn(x):
            return np.argmax(x, axis=1)

        return cls(forward_fn, backward_fn, output_spec)
        

class DefaultInputConverter(BaseInputConverter):
    def __init__(
        self,
        pc: ParameterConfig,
        *,
        scale: bool = True,
        onehot_embed: bool = False,
    ):
        """Do the specific scaling"""
        self._pc = pc
        self.spec = NumpyArraySpec.from_parameter_config(
            pc=pc,
            type_factory=NumpyArraySpecType.default_factory,
        )
        # scaler and onehot_embedder are used for spec with different types,
        # so we can nest them directly
        self.scaler = (
            SpecTransform.scaler(self.spec)
            if scale
            else SpecTransform.identity(self.spec)
        )
        spec = self.scaler.output_spec
        self.onehot_embedder = (
            SpecTransform.onehot_embedder(spec)
            if onehot_embed
            else SpecTransform.identity(spec)
        )
        spec = self.onehot_embedder.output_spec
        self._output_spec = spec

    def convert(self, trials: Sequence[Trial]) -> np.ndarray:
        if not trials:
            return np.zeros((0, self._output_spec.num_dimensions))

        if self.spec.type == NumpyArraySpecType.DOUBLE:
            values = [t.parameters[self._pc.name].value for t in trials]
        elif self.spec.type in (
            NumpyArraySpecType.INTEGER,
            NumpyArraySpecType.CATEGORICAL,
            NumpyArraySpecType.DISCRETE,
        ):
            values = [
                self._pc.feasible_values.index(t.parameters[self._pc.name].value)
                for t in trials
            ]
        else:
            raise NotImplementedError('type {} is not supported'.format(self._pc.type))
        values = np.array(values).reshape(-1, 1)
        
        return self.onehot_embedder.forward_fn(
            self.scaler.forward_fn(values)
        )
    
    def to_parameter_values(self, array: np.ndarray) -> List[ParameterValue]:
        values = self.scaler.backward_fn(self.onehot_embedder.backward_fn(array))
        values = values.flatten()
        if self.spec.type == NumpyArraySpecType.DOUBLE:
            return [ParameterValue(v) for v in values]
        elif self.spec.type in (
            NumpyArraySpecType.INTEGER,
            NumpyArraySpecType.CATEGORICAL,
            NumpyArraySpecType.DISCRETE,
        ):
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
        output_converters: Sequence[BaseOutputConverter],
    ):
        self.input_converter_dict = {
            converter.parameter_config.name: converter for converter in input_converters
        }
        self.output_converter_dict = {
            converter.metric_information.name: converter for converter in output_converters
        }

    @classmethod
    def from_problem(cls, problem: ProblemStatement):
        parameter_configs = problem.search_space.parameter_configs
        objectives = problem.objective.metric_informations

        input_converters = [DefaultInputConverter(pc) for pc in parameter_configs]
        output_converters = [DefaultOutputConverter(m) for m in objectives]

        return cls(input_converters, output_converters)

    def convert(
        self,
        trials: Sequence[Trial]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        return self.to_features(trials), self.to_labels(trials)

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

    def to_trials(
        self,
        features: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray] = None
    ) -> Sequence[Trial]:
        trials = []
        parameters = self._to_parameters(features)
        if labels is not None:
            metrics = self._to_metrics(labels)
        else:
            metrics = [None for _ in range(len(parameters))]
        for p, m in zip(parameters, metrics):
            trials.append(Trial(parameters=p, metrics=m))
        return trials

    def _to_parameters(self, features: Dict[str, np.ndarray]) -> List[ParameterDict]:
        size = next(iter(features.values())).shape[0]
        parameters = [ParameterDict() for _ in range(size)]
        for name, converter in self.input_converter_dict.items():
            parameter_values = converter.to_parameter_values(features[name])
            for p, v in zip(parameters, parameter_values):
                p[name] = v
        return parameters
        
    def _to_metrics(self, labels: Dict[str, np.ndarray]) -> List[MetricDict]:
        size = next(iter(labels.values())).shape[0]
        metrics = [MetricDict() for _ in range(size)]
        for name, converter in self.output_converter_dict.items():
            metric_values = converter.to_metrics(labels[name])
            for m, v in zip(metrics, metric_values):
                m[name] = v
        return metrics

    @property
    def output_spec(self) -> Dict[str, NumpyArraySpec]:
        return {k: v.output_spec for k, v in self.input_converter_dict.values()}

    @property
    def metric_spec(self) -> Dict[str, MetricInformation]:
        return {k: v.metric_information for k, v in self.output_converter_dict.values()}


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
    def from_problem(cls, problem: ProblemStatement):
        converter = cls([], [])
        converter._impl = DefaultTrialConverter.from_problem(problem)
        return converter

    def convert(self, trials: Sequence[Trial]) -> Tuple[np.ndarray, np.ndarray]:
        return self.to_features(trials), self.to_labels(trials)

    def to_features(self, trials: Sequence[Trial]) -> np.ndarray:
        return dict2array(self._impl.to_features(trials))

    def to_labels(self, trials: Sequence[Trial]) -> np.ndarray:
        return dict2array(self._impl.to_labels(trials))

    def to_trials(self, features: np.ndarray, labels: np.ndarray=None) -> Sequence[Trial]:
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
        
        if labels is not None:
            metrics = [MetricDict() for _ in range(size)]
            curr = 0
            for name, converter in self._impl.output_converter_dict.items():
                metric_values = converter.to_metrics(labels[:, curr])
                curr += 1
                for m, v in zip(metrics, metric_values):
                    m[name] = v
        else:
            metrics = [None for _ in range(size)]

        return [Trial(parameters=p, metrics=m) for p, m in zip(parameters, metrics)]

    @property
    def output_spec(self) -> Dict[str, NumpyArraySpec]:
        return {k: v.output_spec for k, v in self._impl.input_converter_dict.values()}

    @property
    def metric_spec(self) -> Dict[str, MetricInformation]:
        return {k: v.metric_information for k, v in self._impl.output_converter_dict.values()}