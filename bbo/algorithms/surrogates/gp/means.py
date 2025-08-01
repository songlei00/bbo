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

import torch
import torch.nn as nn


class ConstantMean(nn.Module):
    def __init__(self):
        super(ConstantMean, self).__init__()
        self.constant = nn.Parameter(torch.zeros(()))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.constant.expand(*X.shape[:-1], 1)
    

class MLPMean(nn.Module):
    def __init__(self, in_d: int, warp_module: nn.Module):
        super(MLPMean, self).__init__()
        self.warp_module = warp_module
        self.mean = nn.Linear(in_d, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mean(self.warp_module(X))
