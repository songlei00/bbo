import enum
from typing import Optional, Dict

from attrs import define, field, validators


class ObjectiveMetricGoal(enum.IntEnum):
    MAXIMIZE = 1
    MINIMIZE = 2

    @property
    def is_maximize(self) -> bool:
        return self == self.MAXIMIZE

    @property
    def is_minimize(self) -> bool:
        return self == self.MINIMIZE
    

def _min_leq_max(instance, _, value):
    if value > instance.max_value:
        raise ValueError('min_value cannot exceed max_value')
    

def _max_geq_min(instance, _, value):
    if value < instance.min_value:
        raise ValueError('min_value cannot exceed max_value')
    

@define
class MetricInformation:
    name: str = field(
        default='',
        validator=validators.instance_of(str),
    )
    goal: ObjectiveMetricGoal = field(
        validator=validators.instance_of(ObjectiveMetricGoal),
        kw_only=True,
    )
    min_value: Optional[float] = field(
        default=float('-inf'),
        converter=lambda x: float(x) if x is not None else float('-inf'),
        validator=validators.optional((
            validators.instance_of(float),
            _min_leq_max,
        )),
        kw_only=True,
    )
    max_value: Optional[float] = field(
        default=float('inf'),
        converter=lambda x: float(x) if x is not None else float('inf'),
        validator=validators.optional((
            validators.instance_of(float),
            _max_geq_min,
        )),
        kw_only=True,
    )

    @property
    def range(self):
        return self.max_value - self.min_value


@define
class Objective:
    _metric_informations: Dict[str, MetricInformation] = field(init=False, factory=dict)

    def get(self, name: str) -> MetricInformation:
        return self._metric_informations[name]

    def add_metric(
        self,
        name: str,
        goal: ObjectiveMetricGoal,
        min_value: float = float('-inf'),
        max_value: float = float('inf'),
    ) -> MetricInformation:
        metric = MetricInformation(
            name=name,
            goal=goal,
            min_value=min_value,
            max_value=max_value,
        )
        self._metric_informations[name] = metric
        return metric

    @property
    def metrics(self):
        return list(self._metric_informations.values())

    @property
    def metric_informations(self):
        return self._metric_informations

    def num_metrics(self, goal: Optional[ObjectiveMetricGoal]=None):
        if goal is None:
            return len(self._metric_informations)
        return [m.goal for m in self._metric_informations.values()].count(goal)
        