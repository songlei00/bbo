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
import json
from typing import Optional, Dict, List, Sequence, Optional

import numpy as np
from attrs import define, field, validators, asdict

from bbo.shared.metadata import Metadata
from bbo.shared.serializable import Serializable
from bbo.shared.trial import Trial
from bbo.algorithms.abstractions import PartiallySerializableDesigner, CompletedTrials, ActiveTrials
from bbo.algorithms.converters.core import DefaultTrialConverter, TrialConverter
from bbo.shared.base_study_config import ProblemStatement
from bbo.utils import attrs_utils, math_utils, json_utils


@define
class Population(Serializable):
    xs: Dict[str, np.ndarray] = field(
        validator=validators.deep_mapping(
            key_validator=validators.instance_of(str),
            value_validator=validators.and_(
                validators.instance_of(np.ndarray),
                attrs_utils.shape_equals(lambda x: (len(x), 1))
            )
        )
    )
    ys: Optional[Dict[str, np.ndarray]] = field(
        validator=validators.optional(validators.deep_mapping(
            key_validator=validators.instance_of(str),
            value_validator=validators.and_(
                validators.instance_of(np.ndarray),
                attrs_utils.shape_equals(lambda x: (len(x), 1))
            )
        )),
        default=None
    )
    ages: Optional[np.ndarray] = field(
        validator=validators.optional(validators.and_(
            validators.instance_of(np.ndarray),
            attrs_utils.shape_equals(lambda x: (len(x),))
        )),
        default=None
    )
    generations: Optional[np.ndarray] = field(
        validator=validators.optional(validators.and_(
            validators.instance_of(np.ndarray),
            attrs_utils.shape_equals(lambda x: (len(x),))
        )),
        default=None
    )
    ids: Optional[np.ndarray] = field(
        validator=validators.optional(validators.and_(
            validators.instance_of(np.ndarray),
            attrs_utils.shape_equals(lambda x: (len(x),))
        )),
        default=None
    )
    def __attrs_post_init__(self):
        if self.ages is None:
            self.ages = np.zeros(len(self), dtype=np.int32)
        if self.generations is None:
            self.generations = np.zeros(len(self), dtype=np.int32)
        if self.ids is None:
            self.ids = np.zeros(len(self), dtype=np.int32)
        self._check_length()

    def _check_length(self):
        for v in self.xs.values():
            assert len(v) == len(self)
        if self.ys is not None:
            for v in self.ys.values():
                assert len(v) == len(self)
        assert len(self.ages) == len(self)
        assert len(self.generations) == len(self)
        assert len(self.ids) == len(self)

    def __getitem__(self, indices: int | slice | List[int]):
        if isinstance(indices, int):
            indices = [indices]
        return Population(
            xs={k: v[indices] for k, v in self.xs.items()},
            ys={k: v[indices] for k, v in self.ys.items()} if self.ys is not None else None,
            ages=self.ages[indices],
            generations=self.generations[indices],
            ids=self.ids[indices]
        )

    def __len__(self):
        return len(next(iter(self.xs.values())))

    def __add__(self, other: 'Population') -> 'Population':
        return Population(
            xs={k: np.concatenate([self.xs[k], other.xs[k]], axis=0) for k in self.xs.keys()},
            ys={k: np.concatenate([self.ys[k], other.ys[k]], axis=0) for k in self.ys.keys()} if self.ys is not None else None,
            ages=np.concatenate([self.ages, other.ages]),
            generations=np.concatenate([self.generations, other.generations]),
            ids=np.concatenate([self.ids, other.ids])
        )

    def __eq__(self, other: 'Population') -> bool:
        return (
            math_utils.eq_dict_of_ndarray(self.xs, other.xs)
            and math_utils.eq_dict_of_ndarray(self.ys, other.ys)
            and np.allclose(self.ages, other.ages)
            and np.allclose(self.generations, other.generations)
            and np.allclose(self.ids, other.ids)
        )
    
    def y_item(self) -> np.ndarray:
        if len(self.ys) > 1:
            ValueError(f'Item method can only be called for single objective (there are {len(self.ys)})')
        return list(self.ys.values())[0]
    
    def increase_ages(self):
        self.ages += 1

    @classmethod
    def recover(cls, metadata: Metadata) -> 'Population':
        encoded = metadata.get('population', default='', cls=str)
        decoded = json.loads(encoded, object_hook=json_utils.numpy_hook)
        return cls(**decoded)

    def dump(self) -> Metadata:
        encoded = json.dumps(asdict(self), cls=json_utils.NumpyEncoder)
        return Metadata({'population': encoded})


class Sampler(abc.ABC):
    @abc.abstractmethod
    def __call__(self, count: int) -> Population:
        pass


class Selection(abc.ABC):
    @abc.abstractmethod
    def __call__(self, population: Population) -> Population:
        pass


class Crossover(abc.ABC):
    @abc.abstractmethod
    def __call__(self, population: Population) -> Population:
        pass


class Mutation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, population: Population) -> Population:
        pass


class Survival(abc.ABC):
    @abc.abstractmethod
    def __call__(self, population: Population) -> Population:
        pass


class PopulationConverter(abc.ABC):
    @abc.abstractmethod
    def to_population(self, trials: Sequence[Trial]) -> Population:
        pass

    @abc.abstractmethod
    def to_trials(self, population: Population) -> List[Trial]:
        pass


@define
class DefaultPopulationConverter(PopulationConverter):
    _problem_statement: ProblemStatement = field(
        validator=validators.instance_of(ProblemStatement)
    )
    _metadata_ns: str = field(
        default='population',
        validator=validators.instance_of(str)
    )
    _trial_converter: TrialConverter = field(init=False)

    def __attrs_post_init__(self):
        self._trial_converter = DefaultTrialConverter.from_study_config(self._problem_statement)

    def to_population(self, trials: Sequence[Trial]) -> Population:
        n_completed = sum(t.is_completed for t in trials)
        if n_completed == 0:
            empty_x = self._trial_converter.to_features([])
            population = Population(xs=empty_x)
            for trial in trials:
                try:
                    ind = Population.recover(trial.metadata.ns(self._metadata_ns))
                    ind.xs = self._trial_converter.to_features([trial])
                except json.decoder.JSONDecodeError:
                    ind = Population(self._trial_converter.to_features([trial]))
                population += ind
        elif n_completed == len(trials):
            empty_x, empty_y = self._trial_converter.to_xy([])
            population = Population(xs=empty_x, ys=empty_y)
            for trial in trials:
                try:
                    ind = Population.recover(trial.metadata.ns(self._metadata_ns))
                    ind.xs, ind.ys = self._trial_converter.to_xy([trial])
                except json.decoder.JSONDecodeError:
                    xs, ys = self._trial_converter.to_xy([trial])
                    ind = Population(xs, ys)
                population += ind
        else:
            raise ValueError('Converter cannot tackle mixed completed and active trials')
        return population

    def to_trials(self, population: Population) -> List[Trial]:
        trials = self._trial_converter.to_trials(population.xs, population.ys)
        for i, trial in enumerate(trials):
            trial.metadata.ns(self._metadata_ns).update(population[i].dump())
        return trials
    
    @property
    def trial_converter(self):
        return self._trial_converter


@define
class CanonicalEvolutionDesigner(PartiallySerializableDesigner, abc.ABC):
    problem_statement: ProblemStatement = field(validator=validators.instance_of(ProblemStatement))
    pop_size: int = field(validator=validators.instance_of(int))
    offspring_size: Optional[int] = field(
        default=None,
        validator=validators.optional(validators.instance_of(int))
    )
    sampling: Sampler = field(validator=validators.instance_of(Sampler), kw_only=True)
    selection: Selection = field(validator=validators.instance_of(Selection), kw_only=True)
    crossover: Optional[Crossover] = field(
        validator=validators.optional(validators.instance_of(Crossover)),
        kw_only=True
    )
    mutation: Optional[Mutation] = field(
        validator=validators.optional(validators.instance_of(Mutation)),
        kw_only=True
    )
    survival: Survival = field(validator=validators.instance_of(Survival), kw_only=True)
    
    _converter: PopulationConverter = field(init=False)
    _population: Population = field(init=False)

    def __attrs_post_init__(self):
        self._converter = DefaultPopulationConverter(self.problem_statement)
        empty_x, empty_y = self._converter.trial_converter.to_xy([])
        self._population = Population(empty_x, empty_y)
        self.offspring_size = self.offspring_size or self.pop_size
        assert self.crossover or self.mutation, 'One of crossover or mutation must be set'

    def suggest(self, count: Optional[int] = None) -> Sequence[Trial]:
        count = count or self.offspring_size
        if len(self._population) < self.pop_size:
            ret = self.sampling(self.pop_size - len(self._population))
            return self._converter.to_trials(ret)

        selected = self.selection(self._population)
        if self.crossover is not None:
            if len(selected) < 2:
                raise ValueError('Crossover requires at least 2 parents')
            selected = self.crossover(selected)
        if self.mutation is not None:
            selected = self.mutation(selected)

        return self._converter.to_trials(selected)
    
    def update(self, completed: CompletedTrials, active: Optional[ActiveTrials] = None):
        completed = completed.trials
        candidates = self._population + self._converter.to_population(completed)
        self._population = self.survival(candidates)
        self._population.increase_ages()

    def load(self, metadata: Metadata):
        self._population = type(self._population).recover(metadata)
    
    def dump(self) -> Metadata:
        return self._population.dump()