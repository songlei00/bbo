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
import datetime
from typing import Union, Optional, Dict, List
from collections import UserDict

from attrs import define, field, validators

from bbo.shared.metadata import Metadata
from bbo.utils import attrs_utils

ParameterValueTypes = Union[str, int, float]


class ParameterType(enum.Enum):
    DOUBLE = 'DOUBLE'
    INTEGER = 'INTEGER'
    CATEGORICAL = 'CATEGORICAL'
    DISCRETE = 'DISCRETE'
    CUSTOM = 'CUSTOM'

    def is_numeric(self) -> bool:
        return self in [self.DOUBLE, self.INTEGER, self.DISCRETE]

    def is_continuous(self) -> bool:
        return self == self.DOUBLE
    
    def _raise_type_error(self, value: ParameterValueTypes):
        raise TypeError(f'Type {self} is not compatible with value: {value}')
    
    def assert_correct_type(self, value: ParameterValueTypes):
        if self.is_numeric() and float(value) != value:
            self._raise_type_error(value)

        if self == ParameterType.CATEGORICAL and (not isinstance(value, (str, bool))):
            self._raise_type_error(value)

        if self == ParameterType.INTEGER and int(value) != value:
            self._raise_type_error(value)


class TrialStatus(enum.Enum):
    UNKNOWN = 'UNKNOWN'
    REQUESTED = 'REQUESTED'
    ACTIVE = 'ACTIVE'
    COMPLETED = 'COMPLETED'
    STOPPING = 'STOPPING'


@define
class Metric:
    value: float = field(
        converter=float,
        validator=validators.instance_of(float)
    )
    std: Optional[float] = field(
        converter=lambda x: float(x) if x else None,
        validator=[
            validators.optional(validators.instance_of(float)),
            validators.optional(attrs_utils.assert_not_negative)
        ],
        default=None
    )


@define
class ParameterValue:
    value: ParameterValueTypes = field(
        init=True,
        validator=validators.instance_of(ParameterValueTypes)
    )

    def cast_as_internal(self, internal_type: ParameterType) -> ParameterValueTypes:
        internal_type.assert_correct_type(self.value)

        if internal_type in (ParameterType.DOUBLE, ParameterType.DISCRETE):
            return self.as_float()
        elif internal_type == ParameterType.INTEGER:
            return self.as_int()
        elif internal_type == ParameterType.CATEGORICAL:
            return self.as_str()
        else:
            raise RuntimeError(f'Unknown type {internal_type}')
        
    def as_float(self):
        return float(self.value)
    
    def as_int(self):
        return int(self.value)
    
    def as_str(self):
        return str(self.value)


class MetricDict(UserDict[str, Metric]):
    def __setitem__(self, key: str, value: Union[Metric, float]):
        if not isinstance(value, Metric):
            value = Metric(value)
        return super().__setitem__(key, value)

    def get_value(self, key: str, default: Optional[float]=None):
        if key in self.data:
            return self.data[key].value
        else:
            return default

    def get_float_dict(self) -> Dict[str, float]:
        return {k: v.value for k, v in self.data.items()}


@define
class Measurement:
    metrics: MetricDict = field(
        factory=MetricDict,
        validator=validators.instance_of(MetricDict),
        converter=MetricDict
    )
    elasped_secs: float = field(
        default=0,
        converter=float,
        validator=attrs_utils.assert_not_negative,
        kw_only=True
    )
    metadata: Metadata = field(
        factory=Metadata,
        validator=validators.instance_of(Metadata),
        kw_only=True
    )


class ParameterDict(UserDict[str, ParameterValue]):
    def __setitem__(self, key: str, value: Union[ParameterValue, ParameterValueTypes]):
        if not isinstance(value, ParameterValue):
            value = ParameterValue(value)
        return super().__setitem__(key, value)
    
    def get_value(self, key: str, default: Optional[ParameterValueTypes]=None):
        if key in self.data:
            return self.data[key].value
        else:
            return default
        
    def get_float_dict(self) -> Dict[str, ParameterValueTypes]:
        return {k: v.value for k, v in self.data.items()}


@define
class Trial:
    parameters: ParameterDict = field(
        factory=ParameterDict,
        validator=validators.instance_of(ParameterDict),
        converter=ParameterDict
    )
    metadata: Metadata = field(
        factory=Metadata,
        validator=validators.instance_of(Metadata),
        kw_only=True
    )
    id: int = field(
        default=0,
        validator=validators.instance_of(int),
        kw_only=True
    )
    measurements: List[Measurement] = field(
        factory=list,
        validator=validators.deep_iterable(
            validators.instance_of(Measurement),
            validators.instance_of(list)
        ),
        kw_only=True
    )
    final_measurement: Optional[Measurement] = field(
        default=None,
        validator=validators.optional(validators.instance_of(Measurement)),
        kw_only=True
    )
    is_requested: bool = field(
        default=False,
        validator=validators.instance_of(bool),
        kw_only=True
    )
    assigned_worker: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        kw_only=True
    )
    stop_reason: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        kw_only=True
    )
    infeasibility_reason: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        kw_only=True
    )
    creation_time: Optional[datetime.datetime] = field(
        factory=datetime.datetime.now,
        validator=validators.optional(validators.instance_of(datetime.datetime)),
        repr=lambda v: v.strftime('%x %X') if v is not None else 'None',
        kw_only=True
    )
    completion_time: Optional[datetime.datetime] = field(
        default=None,
        validator=validators.optional(validators.instance_of(datetime.datetime)),
        repr=lambda v: v.strftime('%x %X') if v is not None else 'None',
        kw_only=True 
    )

    def __attrs_post_init__(self):
        if self.completion_time is None and (
            self.final_measurement is not None or self.infeasible
        ):
            self.completion_time = self.creation_time

    @property
    def duration(self) -> Optional[datetime.timedelta]:
        if self.completion_time is None:
            return None
        else:
            return self.completion_time - self.creation_time

    @property
    def status(self) -> TrialStatus:
        if self.final_measurement is not None or self.infeasible:
            return TrialStatus.COMPLETED
        elif self.stop_reason is not None:
            return TrialStatus.STOPPING
        elif self.is_requested:
            return TrialStatus.REQUESTED
        else:
            return TrialStatus.ACTIVE

    @property
    def infeasible(self):
        return self.infeasibility_reason is not None
    
    @property
    def is_completed(self):
        return self.status == TrialStatus.COMPLETED

    def complete(
        self,
        measurement: Measurement,
        *,
        infeasibility_reason: Optional[str] = None
    ):
        self.__setattr__('final_measurement', measurement)
        self.__setattr__('infeasibility_reason', infeasibility_reason)
        self.completion_time = datetime.datetime.now()