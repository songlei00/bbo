import random
from typing import Optional, Sequence

import numpy as np
from attrs import define, field, validators

from bbo.algorithms.base import Designer
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.parameter_config import ParameterConfig, ParameterType
from bbo.utils.trial import Trial, ParameterDict, ParameterValue


@define
class GridSearchDesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement),
    )
    _shuffle: bool = field(
        default=False,
        validator=validators.instance_of(bool),
        kw_only=True,
    )
    _double_grid_resolution: int = field(
        default=10,
        validator=validators.instance_of(int),
        kw_only=True,
    )

    def __attrs_post_init__(self):
        self._current_idx = 0
        self._grid_values = dict()
        for name, pc in self._problem_statement.search_space.parameter_configs.items():
            self._grid_values[name] = self._get_grid_from_parameter_config(pc)
        
        if self._shuffle:
            self._grid_values = self._shuffle_grid()

    def suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        # Cartesian Product implementation from vizier
        count = count or 1
        parameter_dicts = []
        for idx in range(self._current_idx, self._current_idx+count):
            parameter_dict = ParameterDict()
            temp_idx = idx
            for name, grid in self._grid_values.items():
                p_length = len(grid)
                p_idx = temp_idx % p_length
                parameter_dict[name] = grid[p_idx]
                temp_idx = temp_idx // p_length
            parameter_dicts.append(parameter_dict)

        self._current_idx += count
        return [Trial(pd) for pd in parameter_dicts]

    def update(self, completed: Sequence[Trial]) -> None:
        pass

    def _get_grid_from_parameter_config(
        self,
        pc: ParameterConfig
    ) -> Sequence[ParameterValue]:
        if pc.type == ParameterType.DOUBLE:
            return [
                ParameterValue(v) for v in
                np.linspace(pc.bounds[0], pc.bounds[1], self._double_grid_resolution)
            ]
        elif pc.type in (
            ParameterType.INTEGER,
            ParameterType.DISCRETE,
            ParameterType.CATEGORICAL,
        ):
            return [ParameterValue(v) for v in pc.feasible_values]
        else:
            raise ValueError('Unsupported type: {}'.format(pc.type))

    def _shuffle_grid(self):
        # shuffle the keys. For python3.6 onwards, dict maintains insertion order
        grid_values = list(self._grid_values.items())
        random.shuffle(grid_values)
        grid_values = dict(grid_values)
        # shuffle the values
        for k in grid_values.keys():
            random.shuffle(grid_values[k])
        return grid_values
