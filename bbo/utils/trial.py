import copy
from datetime import datetime
from typing import Union, Optional
from collections import UserDict

from attrs import define, field, validators

from bbo.utils.metric_config import Objective, ObjectiveMetricGoal
from bbo.utils.metadata import Metadata


ParameterValueTypes = Union[str, int, float]


def _std_not_negative(instance, _, value):
    if value is not None and value < 0:
        raise ValueError('std must be positive')


@define
class Metric:
    value: float = field(
        converter=float,
        validator=validators.instance_of(float),
    )
    std: Optional[float] = field(
        converter=lambda x: float(x) if x is not None else x,
        validator=validators.optional([
            validators.instance_of(float),
            _std_not_negative,
        ]),
        default=None,
    )


class MetricDict(UserDict):
    def __setitem__(self, key: str, value: Union[Metric, float]):
        if not isinstance(value, Metric):
            value = Metric(value)
        return super().__setitem__(key, value)

    def get_value(self, key: str, default: Optional[float] = None):
        if key in self.data:
            return self.data[key].value
        else:
            return default


@define
class ParameterValue:
    value: ParameterValueTypes = field(
        validator=validators.instance_of((str, int, float)),
    )


class ParameterDict(UserDict):
    def __setitem__(self, key: str, value: Union[ParameterValue, ParameterValueTypes]) -> None:
        if not isinstance(value, ParameterValue):
            value = ParameterValue(value)
        return super().__setitem__(key, value)

    def get_value(self, key: str, default: Optional[ParameterValueTypes] = None):
        if key in self.data:
            return self.data[key].value
        else:
            return default


@define
class Trial:
    parameters: ParameterDict = field(
        factory=ParameterDict,
        converter=ParameterDict,
        validator=validators.instance_of(ParameterDict),
    )
    metadata: Metadata = field(
        kw_only=True,
        factory=Metadata,
        validator=validators.instance_of(Metadata),
    )
    id: int = field(
        kw_only=True,
        default=0,
        validator=validators.instance_of(int),
    )
    metrics: Optional[MetricDict] = field(
        kw_only=True,
        default=None,
        converter=MetricDict,
        validator=validators.optional(validators.instance_of(MetricDict)),
    )
    # TODO: update the following parameters when calling complete
    creation_time: Optional[datetime] = field(
        kw_only=True,
        default=None,
        validator=validators.optional(validators.instance_of(datetime)),
    )
    completion_time: Optional[datetime] = field(
        kw_only=True,
        default=None,
        validator=validators.optional(validators.instance_of(datetime)),
    )

    def complete(
        self,
        metrics: MetricDict,
    ):
        # Use setattr to run the validator for metrics
        self.__setattr__('metrics', copy.deepcopy(metrics))


def is_better_than(
    objective: Objective,
    trial1: Trial,
    trial2: Trial,
) -> bool:
    is_better = []
    for name, metric_information in objective.metric_informations.items():
        if metric_information.goal == ObjectiveMetricGoal.MAXIMIZE:
            comp = trial1.metrics[name].value > trial2.metrics[name].value
            is_better.append(comp)
        elif metric_information.goal == ObjectiveMetricGoal.MINIMIZE:
            comp = trial1.metrics[name].value < trial2.metrics[name].value
            is_better.append(comp)
        else:
            raise ValueError('Unsupported goal: {}'.format(metric_information.goal))

    if len(is_better) > 1:
        # multi objective
        raise NotImplementedError('Unsupported for multi objective')
    else:
        return is_better[0]
