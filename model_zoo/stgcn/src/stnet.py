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
"""STGCN net"""
import mindspore as ms
from mindspore_gl import Graph
from mindspore_gl.nn.gnn_cell import GNNCell
from mindspore_gl.nn.temporal import STConv

class STGcnNet(GNNCell):
    """ STGCN Net """
    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 hidden_channels_1st: int,
                 out_channels_1st: int,
                 hidden_channels_2nd: int,
                 out_channels_2nd: int,
                 out_channels: int,
                 kernel_size: int,
                 k: int,
                 bias: bool = True):
        super().__init__()
        self.layer0 = STConv(num_nodes, in_channels,
                             hidden_channels_1st,
                             out_channels_1st,
                             kernel_size,
                             k, bias)
        self.layer1 = STConv(num_nodes, out_channels_1st,
                             hidden_channels_2nd,
                             out_channels_2nd,
                             kernel_size,
                             k, bias)
        self.relu = ms.nn.ReLU()
        self.fc = ms.nn.Dense(out_channels_2nd, out_channels)

    def construct(self, x, edge_weight, g: Graph):
        x = self.layer0(x, edge_weight, g)
        x = self.layer1(x, edge_weight, g)
        x = self.relu(x)
        x = self.fc(x)
        return x
