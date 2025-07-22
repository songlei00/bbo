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

from bbo.algorithms.converters.torch_converter import TrialToTorchConverter
from bbo.algorithms.converters.core_test import compare_trials, create_dummy_ps, generate_trials

trial_size = 5
ps, trials, cardinality = create_dummy_ps(trial_size)
active_trials = generate_trials(ps, trial_size, False)


class TestTrialToTorchConverter:
    def test_default(self):
        converter = TrialToTorchConverter.from_study_config(ps)
        x, y = converter.to_xy(trials)
        assert x.double.shape == (trial_size, 1)
        assert x.integer.shape == (trial_size, 1)
        assert x.discrete.shape == (trial_size, 1)
        assert x.categorical.shape == (trial_size, 1)
        assert y.shape == (trial_size, 1)
        ts = converter.to_trials(x, y)
        compare_trials(ts, trials)

    def test_to_trials_without_y(self):
        converter = TrialToTorchConverter.from_study_config(ps)
        x = converter.to_features(active_trials)
        ts = converter.to_trials(x)
        compare_trials(ts, active_trials)