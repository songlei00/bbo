import random
import math

import pytest
import numpy as np

from bbo.shared.trial import Trial, Measurement
from bbo.utils.pandas_utils import (
    flatten_trial,
    trials2df
)


def _generate_trial(num_measurement: int, completed: bool):
    measurements = [
        Measurement(
            metrics={'c': random.random()},
            elapsed_secs=random.random()
        )
        for _ in range(num_measurement)
    ]
    trial = Trial(
        id=1,
        parameters={'a': 1, 'b': 2},
        measurements=measurements
    )
    if completed:
        trial.complete(Measurement(metrics={'c': random.random()}, elapsed_secs=random.random()))
    return trial


def test_flatten_trial():
    trial = _generate_trial(1, True)
    d = flatten_trial(trial)
    assert len(d.keys()) == 9


@pytest.mark.parametrize('size', [100])
def test_trials2df(size):
    trials = [_generate_trial(random.randint(0, 2), bool(random.randint(0, 1))) for _ in range(size)]
    df = trials2df(trials)

    for i in range(size):
        assert df['id'][i] == trials[i].id

        # Parameters
        for name in trials[i].parameters:
            assert math.isclose(df[name][i], trials[i].parameters[name].value, rel_tol=1e-6)

        # Measurements
        for j, measurement in enumerate(trials[i].measurements):
            for name in measurement.metrics:
                assert math.isclose(
                    df[f'measurement{j}_{name}'][i],
                    measurement.metrics[name].value,
                    rel_tol=1e-6
                )
            assert math.isclose(df[f'measurement{j}_elapsed_secs'][i], measurement.elapsed_secs, rel_tol=1e-6)

        # Final measurements
        if trials[i].final_measurement is not None:
            for name in trials[i].final_measurement.metrics:
                assert math.isclose(
                    df[f'final_measurement_{name}'][i],
                    trials[i].final_measurement.metrics[name].value,
                    rel_tol=1e-6
                )
            assert math.isclose(df['final_measurement_elapsed_secs'][i], trials[i].final_measurement.elapsed_secs, rel_tol=1e-6)

        # Creation and completion time
        assert df['creation_time'][i] == trials[i].creation_time
        if trials[i].completion_time is not None:
            assert df['completion_time'][i] == trials[i].completion_time