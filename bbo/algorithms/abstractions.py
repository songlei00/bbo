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
from typing import Optional, Sequence

from attrs import define, field, validators

from bbo.shared.serializable import PartiallySerializable, Serializable
from bbo.shared.trial import Trial, TrialStatus


def check_trial_status(status: TrialStatus):
    def checker(instance, attribute, value: Trial):
        if value.status != status:
            raise ValueError(f"All trial status must be {status}. Given {value.status}")
    return checker


@define(frozen=True)
class CompletedTrials:
    trials: Sequence[Trial] = field(
        converter=tuple,
        validator=validators.deep_iterable(
            validators.and_(validators.instance_of(Trial), check_trial_status(TrialStatus.COMPLETED)),
        )
    )


@define(frozen=True)
class ActiveTrials:
    trials: Sequence[Trial] = field(
        converter=tuple,
        validator=validators.deep_iterable(
            validators.and_(validators.instance_of(Trial), check_trial_status(TrialStatus.ACTIVE)),
        )
    )


class Designer(abc.ABC):
    @abc.abstractmethod
    def suggest(
        self,
        count: Optional[int] = None
    ) -> Sequence[Trial]:
        pass

    @abc.abstractmethod
    def update(
        self,
        completed_trials: CompletedTrials,
        active_trials: Optional[ActiveTrials] = None
    ) -> None:
        pass


class PartiallySerializableDesigner(Designer, PartiallySerializable):
    pass


class SerializableDesigner(Designer, Serializable):
    pass