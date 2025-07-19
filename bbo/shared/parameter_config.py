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

import enum
import math
import copy
import random
from typing import Union, Tuple, List, Optional, Dict, Sequence, Any

from attrs import define, field, validators

from bbo.shared.trial import ParameterType, ParameterValueTypes, ParameterValue
from bbo.utils.frozendict_utils import FrozenDict
from bbo.utils import attrs_utils

BoundsType = Union[Tuple[int, int], Tuple[float, float]]
MonotypeParameterSequence = Union[List[Union[float, int]], List[str]]


class ScaleType(enum.Enum):
    LINEAR = 'LINEAR'
    LOG = 'LOG'
    REVERSE_LOG = 'REVERSE_LOG'
    UNIFORM_DISCRETE = 'UNIFORM_DISCRETE'


def _check_bounds(bounds: BoundsType):
    if len(bounds) != 2:
        raise ValueError(f'Bounds must have length 2. Given {bounds}')
    low, high = bounds[0], bounds[1]
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f'Low and high must be finite. Given ({low}, {high})')
    if low > high:
        raise ValueError(f'Low {low} is greater than high {high}')
    

def _get_default_value(type: ParameterType, default_value: ParameterValueTypes):
    if type in (ParameterType.DOUBLE, ParameterType.DISCRETE) and isinstance(default_value, (float, int)):
        return float(default_value)
    elif type == ParameterType.INTEGER and isinstance(default_value, (float, int)):
        default_int_value = int(default_value)
        if not math.isclose(default_value, default_int_value):
            raise ValueError(f'Default value for INTEGER is not integer. Given {default_value}')
        return default_int_value
    elif type == ParameterType.CATEGORICAL and isinstance(default_value, str):
        return default_value
    elif type == ParameterType.CUSTOM:
        return default_value
    else:
        raise ValueError(f'Default value has incorrect type. Given: type {type}, default_value: {default_value}')


@define(frozen=True)
class ParameterConfig:
    _name: str = field(validator=validators.instance_of(str), kw_only=True)
    _type: ParameterType = field(
        validator=validators.instance_of(ParameterType),
        repr=lambda v: v.name,
        kw_only=True
    )
    _bounds: Optional[BoundsType] = field(
        validator=validators.optional(attrs_utils.assert_bounds),
        kw_only=True
    )
    _feasible_values: Optional[MonotypeParameterSequence] = field(
        validator=validators.optional(validators.deep_iterable(
            validators.instance_of((int, float, str)),
            validators.instance_of(tuple)
        )),
        converter=lambda x: tuple(x) if x is not None else None,
        kw_only=True
    )
    _scale_type: Optional[ScaleType] = field(
        validator=validators.optional(validators.instance_of(ScaleType)),
        kw_only=True
    )
    _default_value: Optional[ParameterValueTypes] = field(
        validator=validators.optional(validators.instance_of(ParameterValueTypes)),
        kw_only=True
    )

    @classmethod
    def factory(
        cls,
        name: str,
        *,
        bounds: BoundsType = None,
        feasible_values: MonotypeParameterSequence = None,
        scale_type: ScaleType = None,
        default_value: Optional[ParameterValueTypes] = None
    ):
        if bool(bounds) and bool(feasible_values):
            raise ValueError('One or none of bounds and feasible values should be provided')
        
        if bounds:
            _check_bounds(bounds)
            if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                inferred_type = ParameterType.INTEGER
                feasible_values = tuple(range(bounds[0], bounds[1] + 1))
            elif isinstance(bounds[0], float) and isinstance(bounds[1], float):
                inferred_type = ParameterType.DOUBLE
            else:
                raise ValueError(f'Bounds must both be ints or doubles, given {bounds}')
        elif feasible_values:
            if len(set(feasible_values)) != len(feasible_values):
                raise ValueError(f'Feasible values cannot have duplicates, given {feasible_values}')
            if all(isinstance(v, (int, float)) for v in feasible_values):
                inferred_type = ParameterType.DISCRETE
                if not all(math.isfinite(v) for v in feasible_values):
                    raise ValueError(f'Feasible values must be finite, given {feasible_values}')
                feasible_values = sorted(feasible_values)
                bounds = (feasible_values[0], feasible_values[-1])
            elif all(isinstance(v, str) for v in feasible_values):
                inferred_type = ParameterType.CATEGORICAL
                feasible_values = sorted(feasible_values)
            else:
                raise ValueError(f'Feasible values must be all numeric or strings, give {feasible_values}')
        else:
            inferred_type = ParameterType.CUSTOM

        if default_value is not None:
            default_value = _get_default_value(inferred_type, default_value)

        pc = cls(
            name=name,
            type=inferred_type,
            bounds=bounds,
            feasible_values=feasible_values,
            scale_type=scale_type,
            default_value=default_value
        )
        if default_value is not None and inferred_type != ParameterType.CUSTOM:
            pc._assert_feasible(default_value)
        return pc
    
    def is_feasible(self, v: Union[ParameterValueTypes, ParameterValue]):
        if isinstance(v, ParameterValue):
            v = v.value
        try:
            self._assert_feasible(v)
        except (TypeError, ValueError):
            return False
        return True
    
    def _assert_feasible(self, v: ParameterValueTypes):
        self._type.assert_correct_type(v)
        
        if self._type == ParameterType.DOUBLE:
            self._assert_bounds(v)
        elif self._type in (ParameterType.CATEGORICAL, ParameterType.DISCRETE, ParameterType.INTEGER):
            self._assert_in_feasible_values(v)
        elif self._type == ParameterType.CUSTOM:
            raise NotImplementedError(f'Feasible check is not implemented for {self._type}')
        else:
            raise RuntimeError(f'Parameter {self._name} has unknown parameter type {self._type}')

    def _assert_bounds(self, v: ParameterValueTypes):
        if v < self._bounds[0] or v > self._bounds[1]:
            raise ValueError(f'Parameter {self._name} has bounds {self._bounds}. Given {v}')

    def _assert_in_feasible_values(self, v: ParameterValueTypes):
        if v not in self._feasible_values:
            raise ValueError(f'Parameter {self._name} has feasible values {self._feasible_values}. Given {v}')

    def continuify(self) -> 'ParameterConfig':
        if self._type == ParameterType.DOUBLE:
            return copy.deepcopy(self)
        elif not self._type.is_numeric():
            raise ValueError(f'Cannot continuify parameter {self._name} with type {self._type}')
        
        scale_type = self._scale_type
        if scale_type == ScaleType.UNIFORM_DISCRETE:
            scale_type = None

        default_value = self._default_value
        if default_value is not None:
            default_value = float(default_value)

        return ParameterConfig.factory(
            self._name, 
            bounds=(float(self._bounds[0]), float(self._bounds[1])),
            scale_type=scale_type,
            default_value=default_value
        )

    def sample(self) -> ParameterValue:
        if self._type in (ParameterType.CATEGORICAL, ParameterType.DISCRETE):
            new_value = random.choice(self._feasible_values)
        elif self._type == ParameterType.DOUBLE:
            new_value = random.uniform(self._bounds[0], self._bounds[1])
        elif self._type == ParameterType.INTEGER:
            new_value = random.randint(self.bounds[0], self.bounds[1])
        elif self._type == ParameterType.CUSTOM:
            raise NotImplementedError(f'Sampling is not implemented for {self._type}')
        return ParameterValue(new_value)

    @property
    def name(self):
        return self._name
    
    @property
    def type(self):
        return self._type
    
    @property
    def bounds(self):
        if self._type in (ParameterType.DOUBLE, ParameterType.INTEGER, ParameterType.DISCRETE):
            return self._bounds
        else:
            raise ValueError(f'Bounds is not available for parameter type {self._type}')

    @property
    def feasible_values(self):
        if self._type in (
            ParameterType.CATEGORICAL,
            ParameterType.DISCRETE,
            ParameterType.INTEGER
        ):
            return self._feasible_values
        else:
            raise ValueError(f'Feasible values is not available for parameter type {self._type}')

    @property
    def scale_type(self):
        return self._scale_type

    @property
    def default_value(self):
        return self._default_value
    
    @property
    def num_feasible_values(self):
        if self._type in (ParameterType.CATEGORICAL, ParameterType.DISCRETE):
            return len(self._feasible_values)
        elif self._type == ParameterType.INTEGER:
            return self._bounds[1] - self._bounds[0] + 1
        elif self._type == ParameterType.DOUBLE:
            return float('inf')
        else:
            raise ValueError(f'Num feasible values is not available for parameter type {self._type}')
    

@define
class SearchSpace:
    _parameter_configs: Dict[str, ParameterConfig] = field(init=False, factory=dict)
    _parent_values: Optional[FrozenDict] = field(
        default=None,
        validator=validators.optional(validators.instance_of(FrozenDict)),
        kw_only=True
    )
    _children: Dict[FrozenDict, 'SearchSpace'] = field(
        factory=dict,
        validator=validators.deep_mapping(
            validators.instance_of(FrozenDict),
            validators.instance_of('SearchSpace')
        ),
        init=False,
        kw_only=True
    )

    def get(self, name) -> ParameterConfig:
        return self._parameter_configs[name]
    
    def pop(self, name) -> ParameterConfig:
        return self._parameter_configs.pop(name)
    
    def _add_parameter(self, pc: ParameterConfig):
        if pc.name in self._parameter_configs:
            raise ValueError(f'Parameter {pc.name} already exists in search space')
        self._parameter_configs[pc.name] = pc

    def add_float_param(
        self,
        name: str,
        min_value: float,
        max_value: float,
        *,
        default_value: float = None,
        scale_type: ScaleType = ScaleType.LINEAR
    ):
        bounds = (float(min_value), float(max_value))
        pc = ParameterConfig.factory(
            name,
            bounds=bounds,
            scale_type=scale_type,
            default_value=default_value
        )
        self._add_parameter(pc)
        return pc
    
    def add_int_param(
        self,
        name: str,
        min_value: int,
        max_value: int,
        *,
        default_value: int = None,
        scale_type: ScaleType = None
    ):
        int_min_value = int(min_value)
        int_max_value = int(max_value)
        if not math.isclose(int_min_value, min_value):
            raise ValueError(f'Min value is not integer. Given {min_value}')
        if not math.isclose(int_max_value, max_value):
            raise ValueError(f'Max value is not integer. Given {max_value}')
        
        bounds = (int_min_value, int_max_value)
        pc = ParameterConfig.factory(
            name,
            bounds=bounds,
            scale_type=scale_type,
            default_value=default_value
        )
        self._add_parameter(pc)
        return pc

    def add_discrete_param(
        self,
        name: str,
        feasible_values: List[Union[float, int]],
        *,
        default_value: Union[float, int] = None,
        scale_type: ScaleType = None
    ):
        for v in feasible_values:
            if not isinstance(v, (float, int)):
                raise ValueError(f'Discrete feasible value must be float or int. Given {v}')
        
        pc = ParameterConfig.factory(
            name,
            feasible_values=feasible_values,
            default_value=default_value,
            scale_type=scale_type
        )
        self._add_parameter(pc)
        return pc
    
    def add_categorical_param(
        self,
        name: str,
        feasible_values: List[str],
        *,
        default_value: str = None,
        scale_type: ScaleType = None
    ):
        for v in feasible_values:
            if not isinstance(v, str):
                raise ValueError(f'Categorical feasible value must be str. Given {v}')
        
        pc = ParameterConfig.factory(
            name,
            feasible_values=feasible_values,
            default_value=default_value,
            scale_type=scale_type
        )
        self._add_parameter(pc)
        return pc
    
    def add_custom_param(
        self,
        name: str,
        *,
        default_value: Any = None
    ):
        pc = ParameterConfig.factory(
            name,
            default_value=default_value
        )
        self._add_parameter(pc)
        return pc

    def subspace(
        self,
        fix_parameters: Dict[str, Union[ParameterValueTypes, Sequence[ParameterValueTypes]]]
    ):
        for name, values in fix_parameters.items():
            if not isinstance(values, tuple):
                values_tuple = tuple(values)
            if name not in self._parameter_configs:
                raise KeyError(f'Parameter {name} not in search space')
            pc = self.get(name)
            values_list = []
            for v in values_tuple:
                v = ParameterValue(v).cast_as_internal(pc.type)
                if not pc.is_feasible(v):
                    raise ValueError(f'Parameter {name} is not feasible. Given {v}')
                values_list.append(v)
            fix_parameters[name] = tuple(values_list)
            
        fix_parameters_frozen = FrozenDict(fix_parameters)
        if fix_parameters_frozen not in self._children:
            self._children[fix_parameters_frozen] = SearchSpace(parent_values=fix_parameters_frozen)
        return self._children[fix_parameters_frozen]

    def sample(self) -> Dict[str, ParameterValue]:
        sample = {}
        for pc in self._parameter_configs.values():
            sample[pc.name] = pc.sample()
        return sample

    @property
    def parameter_configs(self) -> Dict[str, ParameterConfig]:
        return self._parameter_configs
    
    def num_parameters(self, param_type: Optional[ParameterType]=None) -> int:
        if self.is_conditional:
            raise NotImplementedError('Num parameters is not available for conditional search space')
        if param_type is None:
            return len(self._parameter_configs)
        else:
            return [pc.type for pc in self.parameter_configs.values()].count(param_type)

    def is_feasible(self, sample: Dict[str, Union[ParameterValue, ParameterValueTypes]]) -> bool:
        for k, v in sample.items():
            if isinstance(v, ParameterValue):
                sample[k] = v.cast_as_internal(self.get(k).type)
        try:
            self._assert_feasible(sample)
            return True
        except ValueError:
            return False
    
    def _assert_feasible(self, sample: Dict[str, ParameterValueTypes]):
        if self.is_conditional:
            raise NotImplementedError('Is feasible is not available for conditional search space')
        if len(sample) != self.num_parameters():
            raise ValueError(f'Sample size is not equal to num parameters. Given {sample}')
        for k, v in sample.items():
            if k not in self._parameter_configs:
                raise ValueError(f'Parameter {k} not in search space')
            if not self.get(k).is_feasible(v):
                raise ValueError(f'Parameter {k} is not feasible. Given {v}')

    @property
    def is_conditional(self) -> bool:
        return len(self._children) > 0

    @property
    def children(self):
        return self._children