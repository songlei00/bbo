import logging
from collections import defaultdict
from typing import Optional, Sequence, Dict

import numpy as np
import pandas as pd
from attrs import define, field, validators

from bbo.algorithms.base import Designer
from bbo.utils.problem_statement import ProblemStatement
from bbo.utils.parameter_config import ParameterType
from bbo.utils.metric_config import ObjectiveMetricGoal
from bbo.utils.trial import Trial, ParameterDict

logger = logging.getLogger(__file__)

try:
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
except ImportError as e:
    logger.warning('Import hebo error: {}'.format(e))


@define
class HEBODesigner(Designer):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _impl_kwargs: Dict = field(factory=dict, kw_only=True)
    _impl = field(init=False)

    def __attrs_post_init__(self):
        space = self._create_problem()
        self._impl = HEBO(space, **self._impl_kwargs)

    def _create_problem(self):
        space_config = list()
        for name, pc in self._problem_statement.search_space.parameter_configs.items():
            if pc.type == ParameterType.DOUBLE:
                space_config.append({'name': name, 'type': 'num', 'lb': pc.bounds[0], 'ub': pc.bounds[1]})
            elif pc.type == ParameterType.INTEGER:
                space_config.append({'name': name, 'type': 'int', 'lb': pc.bounds[0], 'ub': pc.bounds[1]})
            elif pc.type in [ParameterType.CATEGORICAL, ParameterType.DISCRETE]:
                space_config.append({'name': name, 'type': 'cat', 'categories': pc.feasible_values})
            else:
                raise NotImplementedError
        return DesignSpace().parse(space_config)
    
    def _suggest(self, count: Optional[int]=None) -> Sequence[Trial]:
        count = count or 1
        df = self._impl.suggest(n_suggestions=count)
        suggestions = list()
        for i in range(count):
            parameters = ParameterDict()
            for name in df.columns:
                parameters[name] = df.iloc[i][name]
            trial = Trial(parameters)
            suggestions.append(trial)
        return suggestions

    def _update(self, completed: Sequence[Trial]) -> None:
        param_dict, m_list = defaultdict(list), []
        for c in completed:
            for name, p in c.parameters.items():
                param_dict[name].append(p.value)
            for name, m in c.metrics.items():
                m_list.append(m.value)
        X_df = pd.DataFrame(param_dict)
        Y_np = np.array(m_list).reshape(-1, 1)
        if self._problem_statement.objective.item().goal == ObjectiveMetricGoal.MAXIMIZE:
            Y_np = - Y_np
        self._impl.observe(X_df, Y_np)