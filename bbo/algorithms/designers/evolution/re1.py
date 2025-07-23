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

import json
from collections import deque
from typing import Optional, Sequence

import numpy as np
from attrs import define, field, validators

from bbo.shared.base_study_config import ProblemStatement, ObjectiveMetricGoal
from bbo.algorithms.abstractions import PartiallySerializableDesigner, CompletedTrials, ActiveTrials
from bbo.algorithms.designers.random import RandomDesigner
from bbo.algorithms.converters.core import DefaultTrialConverter, TrialConverter, NumpyArraySpecType
from bbo.shared.trial import Trial
from bbo.shared.metadata import Metadata
from bbo.utils import get_rng, json_utils


def tournament_selection(
    population: Sequence[Trial],
    tournament_size: int,
    goal: ObjectiveMetricGoal,
    rng: np.random.Generator
) -> Trial:
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    candidates = [population[i] for i in indices]
    ys = [list(cand.final_measurement.metrics.values())[0].value for cand in candidates]
    if goal == ObjectiveMetricGoal.MAXIMIZE:
        i = np.argmax(ys)
    else:
        i = np.argmin(ys)
    winner = candidates[i]
    return winner


def random_mutation(
    trial: Trial,
    converter: TrialConverter,
    k: int,
    rng: np.random.Generator
) -> Trial:
    output_specs = converter.output_specs
    keys = rng.choice(list(output_specs.keys()), size=k, replace=False)
    sample = converter.to_features([trial])
    for k in keys:
        output_spec = output_specs[k]
        lb, ub = output_spec.bounds
        shape = (1, 1)
        if output_spec.type == NumpyArraySpecType.DOUBLE:
            sample[k] = rng.random(shape, output_spec.dtype) * (ub - lb) + lb
        elif output_spec.type in (
            NumpyArraySpecType.INTEGER,
            NumpyArraySpecType.DISCRETE,
            NumpyArraySpecType.CATEGORICAL
        ):
            if lb != ub:
                while True:
                    sampled_v = rng.integers(lb, ub+1, shape, output_spec.dtype)
                    if sampled_v != sample[k]:
                        sample[k] = sampled_v
                        break
    ret = [Trial(p) for p in converter.to_parameters(sample)]
    assert len(ret) == 1
    return ret[0]


@define
class RegularizedEvolutionDesigner1(PartiallySerializableDesigner):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement),
    )
    _pop_size: int = field(default=25, validator=validators.instance_of(int))
    _tournament_size: int = field(default=5, validator=validators.instance_of(int))
    _num_mutation: int = field(default=1, validator=validators.instance_of(int))
    _seed_or_rng: np.random.Generator | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((np.random.Generator, int)))
    )

    def __attrs_post_init__(self):
        self._rng = get_rng(self._seed_or_rng)
        self._init_designer = RandomDesigner(self._problem_statement, self._rng)
        self._converter = DefaultTrialConverter.from_study_config(self._problem_statement)
        self._population = deque(maxlen=self._pop_size)

    def suggest(self, count: Optional[int] = None) -> Sequence[Trial]:
        count = count or 1
        if len(self._population) < self._pop_size:
            ret = self._init_designer.suggest(count)
        else:
            count = count or 1
            ret = []
            for _ in range(count):
                parent = tournament_selection(
                    self._population,
                    self._tournament_size,
                    self._problem_statement.metric_information_item().goal,
                    self._rng
                )
                child = random_mutation(parent, self._converter, self._num_mutation, self._rng)
                ret.append(child)
        return ret
    
    def update(self, completed_trials: CompletedTrials, active_trials: Optional[ActiveTrials] = None):
        self._population.extend(completed_trials.trials)

    def load(self, metadata: Metadata):
        self._rng = np.random.Generator(np.random.PCG64())
        self._rng.bit_generator.state = json.loads(metadata['rng'])
        self._population = deque(
            json.loads(metadata['population'], object_hook=json_utils.trial_hook),
            maxlen=self._pop_size
        )
        self._init_designer._rng = self._rng

    def dump(self) -> Metadata:
        rng_encoded = json.dumps(self._rng.bit_generator.state)
        population_encoded = json.dumps(list(self._population), cls=json_utils.TrialEncoder)
        return Metadata({'population': population_encoded, 'rng': rng_encoded})