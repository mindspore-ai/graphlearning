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
"""graphsage Net"""
import math
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common.initializer import XavierUniform

from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell


class SAGEConv(GNNCell):
    """graphsage conv"""

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 aggregator_type: str = "pool",
                 feat_drop=0.6,
                 bias=True,
                 norm=None,
                 activation: ms.nn.Cell = None):
        super().__init__()
        self.in_feat_size = in_feat_size
        self.out_feat_size = out_feat_size
        self.agg_type = aggregator_type
        self.norm = norm
        self.activation = activation
        self.feat_drop = ms.nn.Dropout(feat_drop)
        self.dense_neigh = ms.nn.Dense(self.in_feat_size, self.out_feat_size, has_bias=False,
                                       weight_init=XavierUniform(math.sqrt(2)))
        if bias:
            self.bias = ms.Parameter(ms.ops.Zeros()(self.out_feat_size, ms.float32))
        else:
            self.bias = None
        if self.agg_type == "pool":
            self.fc_pool = ms.nn.Dense(self.in_feat_size, self.in_feat_size)
        elif self.agg_type == "lstm":
            self.lstm = ms.nn.LSTM(self.in_feat_size, self.in_feat_size, batch_first=True)
        self.dense_self = ms.nn.Dense(self.in_feat_size, self.out_feat_size, has_bias=False,
                                      weight_init=XavierUniform(math.sqrt(2)))

    def construct(self, x, edge_weight, g: Graph):
        """graphsage conv forward"""
        node_feat = self.feat_drop(x)
        g.set_edge_attr({"w": edge_weight})
        g.set_vertex_attr({"h": ms.ops.ReLU()(self.fc_pool(node_feat))})
        for v in g.dst_vertex:
            v.rst = g.sum([u.h for u in v.innbs])
            v.rst = self.dense_neigh(v.rst)
        ret = self.dense_self(node_feat) + [v.rst for v in g.dst_vertex]
        if self.bias is not None:
            ret = ret + self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        if self.norm is not None:
            ret = self.norm(self.ret)
        return ret


class SAGENet(Cell):
    """graphsage net"""

    def __init__(self, in_feat_size, hidden_feat_size, out_feat_size):
        super().__init__()
        self.num_layers = 2

        self.layer1 = SAGEConv(in_feat_size, hidden_feat_size)
        self.layer2 = SAGEConv(hidden_feat_size, out_feat_size)
        self.activation = ms.nn.ELU()
        self.dropout = ms.nn.Dropout(0.5)

    def construct(self, node_feat, layered_edges_0, layered_edges_1):
        """graphsage net forward"""
        node_feat = self.layer1(node_feat, 0, layered_edges_1[0], layered_edges_1[1], 1, 1)
        node_feat = self.activation(node_feat)
        node_feat = self.dropout(node_feat)

        ret = self.layer2(node_feat, 0, layered_edges_0[0], layered_edges_0[1], 1, 1)
        return ret
