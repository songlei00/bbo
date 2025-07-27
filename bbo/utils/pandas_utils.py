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

from typing import Sequence
from collections import defaultdict

import pandas as pd

from bbo.shared.trial import Trial


def flatten_trial(trial: Trial) -> dict:
    # TODO: add metadata information
    d = dict()
    d['id'] = trial.id

    # Parameters
    for name in trial.parameters:
        d[name] = trial.parameters[name].value

    # Measurements
    for i, measurement in enumerate(trial.measurements):
        for name in measurement.metrics:
            d[f'measurement{i}_{name}'] = measurement.metrics[name].value
        d[f'measurement{i}_elapsed_secs'] = measurement.elapsed_secs

    # Final measurements
    if trial.final_measurement is not None:
        for name in trial.final_measurement.metrics:
            d[f'final_measurement_{name}'] = trial.final_measurement.metrics[name].value
        d['final_measurement_elapsed_secs'] = trial.final_measurement.elapsed_secs

    # Creation and completion time
    d['creation_time'] = trial.creation_time
    d['completion_time'] = trial.completion_time
    return d


def trials2df(trials: Sequence[Trial]) -> pd.DataFrame:
    df_dict = defaultdict(list)
    for i, trial in enumerate(trials):
        d = flatten_trial(trial)
        for k in d:
            if len(df_dict[k]) != i:
                df_dict[k].extend([None] * (i - len(df_dict[k])))
            df_dict[k].append(d[k])

    for k in df_dict:
        if len(df_dict[k]) < len(trials):
            df_dict[k].extend([None] * (len(trials) - len(df_dict[k])))

    return pd.DataFrame(df_dict)