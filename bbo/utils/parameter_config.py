import math
import enum
import copy
import random
from typing import Optional, Union, Tuple, Dict, List, Sequence

from attrs import define, field, validators

from bbo.utils.trial import ParameterValue, ParameterDict


MonotypeParameterSequence = Union[Sequence[Union[int, float]], Sequence[str]]
BoundTuple = Union[Tuple[float, float], Tuple[int, int]]


class ParameterType(enum.Enum):
    DOUBLE = 'DOUBLE'
    INTEGER = 'INTEGER'
    CATEGORICAL = 'CATEGORICAL'
    DISCRETE = 'DISCRETE'

    def is_numeric(self) -> bool:
        return self in [self.DOUBLE, self.INTEGER, self.DISCRETE]

    def is_continuous(self) -> bool:
        return self in [self.DOUBLE]


class ScaleType(enum.Enum):
    LINEAR = 'LINEAR'
    LOG = 'LOG'

    def is_nonlinear(self) -> bool:
        return self in [self.LOG]
    

def _validate_bounds(bounds: BoundTuple):
    if len(bounds) != 2:
        raise ValueError('Bounds must have length 2')
    if not all(math.isfinite(v) for v in bounds):
        raise ValueError('Bounds must be finite')
    if bounds[0] > bounds[1]:
        raise ValueError('Lower cannot be greater than upper')


def _get_feasible_points_and_bounds(
    feasible_values: Sequence[float],
) -> Tuple[List[float], BoundTuple]:
    if not all(math.isfinite(v) for v in feasible_values):
        raise ValueError(
            'Feasible values must all be finite'
        )

    feasible_points = list(sorted(feasible_values))
    bounds = (feasible_points[0], feasible_points[-1])
    return feasible_points, bounds


def _get_categories(categories: Sequence[str]) -> List[str]:
    return list(sorted(categories))


def _get_default_value(
    param_type: ParameterType, default_value: Union[float, int, str],
) -> Union[float, int, str]:
    if param_type == ParameterType.INTEGER:
        default_int_value = round(default_value)
        if not math.isclose(default_value, default_int_value):
            raise ValueError(
                'default_value for an INTEGER parameter should be an integer'
            )
    return default_value


@define
class ParameterConfig:
    """
    Please use ParameterConfig.factory() to create an instance instead of calling the constructor directly
    """
    _name: str = field(
        validator=validators.instance_of(str), 
        kw_only=True,
    )
    _type: ParameterType = field(
        validator=validators.instance_of(ParameterType),
        kw_only=True,
    )
    _bounds: Optional[Union[Tuple[int, int], Union[float, float]]] = field(
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of((int, float)),
                iterable_validator=validators.instance_of(tuple),
            )
        ),
        kw_only=True,
    )
    _feasible_values: Optional[MonotypeParameterSequence] = field(
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of((int, float, str)),
                iterable_validator=validators.instance_of((list, tuple)),
            )
        ),
        kw_only=True,
    )
    _scale_type: Optional[ScaleType] = field(
        validator=validators.optional(
            validators.instance_of(ScaleType)
        ),
        kw_only=True,
    )
    _default_value: Optional[Union[float, int, str]] = field(
        validator=validators.optional(
            validators.instance_of((float, int, str))
        ),
        kw_only=True,
    )

    @classmethod
    def factory(
        cls,
        name: str,
        *,
        bounds: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
        feasible_values: Optional[MonotypeParameterSequence] = None,
        scale_type: Optional[ScaleType] = None,
        default_value: Optional[Union[float, int, str]] = None,
    ):
        if bool(feasible_values) == bool(bounds):
            raise ValueError(
                'Exactly one of "feasible_values" or "bounds" must be provided'
            )

        if feasible_values:
            if len(set(feasible_values)) != len(feasible_values):
                raise ValueError(
                    'Feasible values cannot have deplicates'
                )

            if all(isinstance(v, (float, int)) for v in feasible_values):
                inferred_type = ParameterType.DISCRETE
                feasible_values, bounds = _get_feasible_points_and_bounds(feasible_values)
            elif all(isinstance(v, str) for v in feasible_values):
                inferred_type = ParameterType.CATEGORICAL
                feasible_values = _get_categories(feasible_values)
            else:
                raise ValueError(
                    'Feasible values must all be numeric or strings'
                )
        else: # bounds are specified
            _validate_bounds(bounds)
            if all(isinstance(v, int) for v in bounds):
                inferred_type = ParameterType.INTEGER
            elif all(isinstance(v, float) for v in bounds):
                inferred_type = ParameterType.DOUBLE
            else:
                raise ValueError('Bounds must both be integers or doubles')

        if default_value is not None:
            default_value = _get_default_value(inferred_type, default_value)

        pc = cls(
            name=name,
            type=inferred_type,
            bounds=bounds,
            feasible_values=feasible_values,
            scale_type=scale_type,
            default_value=default_value,
        )

        return pc

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> ParameterType:
        return self._type

    @property
    def bounds(self) -> BoundTuple:
        if self.type == ParameterType.CATEGORICAL:
            raise ValueError(
                'Accessing bounds of a categorical parameter'
            )
        return self._bounds
    
    @property
    def feasible_values(self) -> MonotypeParameterSequence:
        if self.type in (ParameterType.CATEGORICAL, ParameterType.DISCRETE):
            return copy.copy(self._feasible_values)
        elif self.type == ParameterType.INTEGER:
            return list(range(self.bounds[0], self.bounds[1]+1))
        else:
            raise ValueError('feasible_values is invalid')

    @property
    def scale_type(self) -> ScaleType:
        return self._scale_type

    @property
    def default_value(self) -> Union[float, int, str]:
        return self._default_value

    @property
    def num_feasible_values(self) -> Union[float, int]:
        if self.type == ParameterType.DOUBLE:
            return float('inf')
        elif self.type == ParameterType.INTEGER:
            return self.bounds[1] - self.bounds[0] + 1
        else:
            return len(self._feasible_values)

    def contains(self, value: Union[float, int, str]) -> bool:
        if self.type in (ParameterType.DOUBLE, ParameterType.INTEGER):
            return self.bounds[0] <= value <= self.bounds[1]
        elif self.type in (ParameterType.DISCRETE, ParameterType.CATEGORICAL):
            return value in self._feasible_values

    def sample(self) -> ParameterValue:
        if self.type in (ParameterType.CATEGORICAL, ParameterType.DISCRETE):
            new_value = random.choice(self._feasible_values)
        elif self.type == ParameterType.DOUBLE:
            new_value = random.uniform(self._bounds[0], self._bounds[1])
        elif self.type == ParameterType.INTEGER:
            new_value = random.randint(self.bounds[0], self.bounds[1])
        return ParameterValue(new_value)


@define
class SearchSpace:
    _parameter_configs: Dict[str, ParameterConfig] = field(init=False, factory=dict)

    def get(self, name: str) -> ParameterConfig:
        return self._parameter_configs[name]
    
    def pop(self, name: str) -> ParameterConfig:
        return self._parameter_configs.pop(name)

    def _add_param(
        self, 
        parameter_config: ParameterConfig,
        *,
        replace: bool = False,
    ) -> ParameterConfig:
        name = parameter_config.name
        if name in self._parameter_configs and not replace:
            raise ValueError('Duplicate name')
        self._parameter_configs[name] = parameter_config
        return parameter_config

    def add_float_param(
        self,
        name: str,
        min_value: float,
        max_value: float,
        *,
        default_value: Optional[float] = None,
        scale_type: Optional[ScaleType] = None,
    ) -> ParameterConfig:
        bounds = (float(min_value), float(max_value))
        pc = ParameterConfig.factory(
            name=name,
            bounds=bounds,
            scale_type=scale_type,
            default_value=default_value,
        )
        return self._add_param(pc)

    def add_int_param(
        self,
        name: str,
        min_value: int,
        max_value: int,
        *,
        default_value: Optional[int] = None,
        scale_type: Optional[ScaleType] = None,
    ) -> ParameterConfig:
        int_min_value = int(min_value)
        int_max_value = int(max_value)
        if not math.isclose(min_value, int_min_value) or not math.isclose(max_value, int_max_value):
            raise ValueError(
                'min_value for INTEGER parameter should be an integer'
            )
        bounds = (int_min_value, int_max_value)
        pc = ParameterConfig.factory(
            name=name,
            bounds=bounds,
            scale_type=scale_type,
            default_value=default_value,
        )
        return self._add_param(pc)

    def add_discrete_param(
        self,
        name: str,
        feasible_values: Sequence[Union[float, int]],
        *,
        default_value: Optional[Union[float, int]] = None,
        scale_type: Optional[ScaleType] = None,
    ) -> ParameterType:
        if not all(isinstance(v, (float, int)) for v in feasible_values):
            raise ValueError('feasible_values must be float or int')

        pc = ParameterConfig.factory(
            name=name,
            feasible_values=sorted(feasible_values),
            scale_type=scale_type,
            default_value=default_value,
        )
        return self._add_param(pc)

    def add_categorical_param(
        self,
        name: str,
        feasible_values: Sequence[str],
        *,
        default_value: Optional[str] = None,
        scale_type: Optional[ScaleType] = None,
    ) -> ParameterConfig:
        if not all(isinstance(v, str) for v in feasible_values):
            raise ValueError('feasible_values must be strings')

        pc = ParameterConfig.factory(
            name=name,
            feasible_values=feasible_values,
            scale_type=scale_type,
            default_value=default_value,
        )
        return self._add_param(pc)

    def sample(self) -> ParameterDict:
        return {k: v.sample() for k, v in self._parameter_configs.items()}

    @property
    def parameters(self) -> List[ParameterConfig]:
        return list(self._parameter_configs.values())

    @property
    def parameter_configs(self) -> Dict[str, ParameterConfig]:
        return self._parameter_configs

    def num_parameters(self, param_type: Optional[ParameterType]=None):
        if param_type is None:
            return len(self._parameter_configs)
        return [pc.type for pc in self.parameters].count(param_type)