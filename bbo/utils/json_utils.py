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

import datetime
import copy

import json
import numpy as np
from attrs import asdict

from bbo.shared.trial import Trial, ParameterDict, ParameterValue, MetricDict, Metric, Measurement
from bbo.shared.metadata import Metadata, Namespace


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__ndarray__': True,
                'value': obj.tolist(),
                'dtype': obj.dtype.name,
                'shape': obj.shape}
        return super().default(obj)


def numpy_hook(obj):
    if '__ndarray__' in obj:
        return np.array(obj['value'], dtype=obj['dtype']).reshape(obj['shape'])
    return obj


class TrialEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Trial, ParameterValue, Metric, Measurement)):
            d = asdict(obj, recurse=False)
            d |= {f'__{obj.__class__.__name__.lower()}__': True}
            return d
        elif isinstance(obj, (ParameterDict, MetricDict)):
            d = copy.deepcopy(obj.data)
            d |= {f'__{obj.__class__.__name__.lower()}__': True}
            return d
        elif isinstance(obj, Metadata):
            d = dict()
            for ns, k, v in obj.all_items():
                d[ns.encode()] = (k, v)
            d |= {'__metadata__': True}
            return d
        elif isinstance(obj, datetime.datetime):
            return {'__datetime__': True, 'value': obj.isoformat()}
        return super().default(obj)
    

def trial_hook(obj):
    if '__trial__' in obj:
        del obj['__trial__']
        return Trial(**obj)
    elif '__parametervalue__' in obj:
        del obj['__parametervalue__']
        return ParameterValue(**obj)
    elif '__metric__' in obj:
        del obj['__metric__']
        return Metric(**obj)
    elif '__measurement__' in obj:
        del obj['__measurement__']
        return Measurement(**obj)
    elif '__parameterdict__' in obj:
        del obj['__parameterdict__']
        return ParameterDict(obj)
    elif '__metricdict__' in obj:
        del obj['__metricdict__']
        return MetricDict(obj)
    elif '__metadata__' in obj:
        del obj['__metadata__']
        m = Metadata()
        for ns, kv_pair in obj.items():
            k, v = kv_pair
            m.abs_ns(Namespace.decode(ns))[k] = v
        return m
    elif '__datetime__' in obj:
        del obj['__datetime__']
        return datetime.datetime.fromisoformat(obj['value'])
    return obj