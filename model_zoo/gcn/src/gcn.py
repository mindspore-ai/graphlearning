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
"""gcn"""
import mindspore as ms

from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import GCNConv


class GCNNet(GNNCell):
    """ GCN Net """

    def __init__(self,
                 data_feat_size: int,
                 hidden_dim_size: int,
                 n_classes: int,
                 dropout: float,
                 activation: ms.nn.Cell = None):
        super().__init__()
        self.layer0 = GCNConv(data_feat_size, hidden_dim_size, activation(), dropout)
        self.layer1 = GCNConv(hidden_dim_size, n_classes, None, dropout)

    def construct(self, x, in_deg, out_deg, g: Graph):
        """GCN Net forward"""
        x = self.layer0(x, in_deg, out_deg, g)
        x = self.layer1(x, in_deg, out_deg, g)
        return x
