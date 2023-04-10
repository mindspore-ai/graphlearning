# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""appnp"""
from typing import List

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform

from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import APPNPConv


class APPNPNet(GNNCell):
    """APPNP Net"""

    def __init__(self,
                 in_feats: List[int],
                 hidden_dim: int,
                 n_classes: int,
                 feat_dropout: float,
                 edge_dropout: float,
                 alpha,
                 k,
                 activation: ms.nn.Cell = None):
        super().__init__()
        self.fc0 = nn.Dense(in_feats, hidden_dim, weight_init=XavierUniform())
        self.fc1 = nn.Dense(hidden_dim, n_classes, weight_init=XavierUniform())
        self.act = activation()
        self.feat_drop = nn.Dropout(p=feat_dropout)
        self.propagate = APPNPConv(k, alpha, edge_dropout)

    def construct(self, x, in_deg, out_deg, g: Graph):
        """APPNP Net forward"""
        x = self.feat_drop(x)
        x = self.act(self.fc0(x))
        x = self.fc1(self.feat_drop(x))
        x = self.propagate(x, in_deg, out_deg, g)
        return x
