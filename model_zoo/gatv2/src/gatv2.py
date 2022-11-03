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
"""gatv2"""
from typing import List

import mindspore as ms

from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import GATv2Conv


class GatV2Net(GNNCell):
    """GATv2 Net"""

    def __init__(self,
                 num_layers: int,
                 data_feat_size: int,
                 hidden_dim_size: int,
                 n_classes: int,
                 heads: List[int],
                 input_drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 leaky_relu_slope: float = 0.2,
                 add_norm: bool = True,
                 activation: ms.nn.Cell = None):
        super().__init__()
        self.layer0 = GATv2Conv(data_feat_size, hidden_dim_size, heads[0], input_drop_out_rate, attn_drop_out_rate,
                                leaky_relu_slope, activation(), add_norm)
        self.mid_layers = []
        for i in range(0, num_layers):
            self.mid_layers.append(GATv2Conv(hidden_dim_size * heads[i], n_classes, heads[i + 1], input_drop_out_rate,
                                             attn_drop_out_rate, leaky_relu_slope, None, add_norm))

    def construct(self, x, g: Graph):
        """GATv2 Net forward"""
        x = self.layer0(x, g)
        for layer in self.mid_layers:
            x = layer(x, g)
        return x
