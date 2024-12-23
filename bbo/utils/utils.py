import time
import logging
from typing import Sequence
from datetime import datetime

import pandas as pd

from bbo.utils.trial import Trial, ParameterDict, MetricDict
from bbo.utils.problem_statement import ProblemStatement

logger = logging.getLogger(__name__)


def timer_wrapper(func):
    def wrapper(*args, **kwargs):
        st = time.monotonic()
        ret = func(*args, **kwargs)
        et = time.monotonic()
        logger.info('func: ' + func.__name__ + ', time: {} sec'.format(et - st))
        return ret
    return wrapper

def trials2df(trials: Sequence[Trial]):
    df = dict()
    for name in trials[0].parameters:
        values = [t.parameters[name].value for t in trials]
        df[name] = values
    for obj_name in trials[0].metrics:
        obj_values = [t.metrics[obj_name].value for t in trials]
        df[obj_name] = obj_values
    df = pd.DataFrame(df)
    return df

def df2trials(df: pd.DataFrame, problem_statement: ProblemStatement):
    sp = problem_statement.search_space
    obj = problem_statement.objective
    parameter_dict = [ParameterDict() for _ in range(len(df))]
    metric_dict = [MetricDict() for _ in range(len(df))]
    for name in sp.parameter_configs:
        for i, v in enumerate(df[name]):
            parameter_dict[i][name] = v
    for name in obj.metric_informations:
        for i, v in enumerate(df[name]):
            metric_dict[i][name] = v

    trials = [Trial(parameters=p, metrics=m) for p, m in zip(parameter_dict, metric_dict)]
    return trials

def now_time_str():
    return datetime.now().strftime('%y%m%d%H%M%S')