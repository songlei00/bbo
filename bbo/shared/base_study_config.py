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
from collections import UserList

from attrs import define, field, validators

from bbo.shared.metadata import Metadata
from bbo.shared.parameter_config import SearchSpace


class ObjectiveMetricGoal(enum.Enum):
    MAXIMIZE = 1
    MINIMIZE = 2

    @property
    def is_maximize(self) -> bool:
        return self == self.MAXIMIZE

    @property
    def is_minimize(self) -> bool:
        return self == self.MINIMIZE


@define
class MetricInformation:
    name: str = field(
        validator=validators.instance_of(str)
    )
    goal: ObjectiveMetricGoal = field(
        validator=validators.instance_of(ObjectiveMetricGoal)
    )
    min_value: float = field(
        default=float('-inf'),
        converter=float,
        validator=validators.instance_of(float),
        kw_only=True
    )
    max_value: float = field(
        default=float('inf'),
        converter=float,
        validator=validators.instance_of(float),
        kw_only=True
    )
    
    @property
    def range(self):
        return self.max_value - self.min_value


class MetricInformationList(UserList[MetricInformation]):
    def append(self, item: MetricInformation):
        for i in self.data:
            if i.name == item.name:
                raise ValueError(f'Duplicate metric name {item.name}')
        self.data.append(item)


@define
class ProblemStatement:
    search_space: SearchSpace = field(
        factory=SearchSpace,
        validator=validators.instance_of(SearchSpace)
    )
    metric_information: MetricInformationList = field(
        factory=MetricInformationList,
        converter=MetricInformationList,
        validator=validators.instance_of(MetricInformationList)
    )
    metadata: Metadata = field(
        factory=Metadata,
        validator=validators.instance_of(Metadata),
        kw_only=True
    )

    def metric_information_item(self) -> MetricInformation:
        if len(self.metric_information) != 1:
            raise ValueError(f'Item method can only be called for single objective (there are {len(self.metric_information)})')
        return self.metric_information[0]

    @property
    def is_single_objective(self) -> bool:
        return len(self.metric_information) == 1