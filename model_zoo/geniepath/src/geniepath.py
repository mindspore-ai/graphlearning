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
"""Geniepath"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
from mindspore_gl.nn import GATConv


class GeniePathConv(GNNCell):
    """GeniePath Conv"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_attn_head=1, residual=False):
        super(GeniePathConv, self).__init__()
        self.residual = residual
        self.breadth = GATConv(input_dim, hidden_dim, num_attn_head=num_attn_head)
        self.depth = nn.LSTM(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        self.expand = ops.ExpandDims()

    def construct(self, x, h, c, g: Graph):
        """construct function"""
        x = self.breadth(x, g)
        x = self.tanh(x)
        x = self.expand(x, 0)
        x, (h, c) = self.depth(x, (h, c))
        x = x[0]
        return x, (h, c)


class GeniePath(GNNCell):
    """GeniePath"""

    def __init__(self, input_dim, output_dim, hidden_dim=16, num_layers=2, num_attn_head=1, residual=False):
        super(GeniePath, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense1 = nn.Dense(input_dim, hidden_dim)
        self.dense2 = nn.Dense(hidden_dim, output_dim)
        self.layers = nn.CellList()
        for _ in range(num_layers):
            self.layers.append(GeniePathConv(hidden_dim, hidden_dim, hidden_dim, num_attn_head=num_attn_head,
                                             residual=residual))

    def construct(self, x, g: Graph):
        """construct function"""
        h = ops.Zeros()((1, ops.Shape()(x)[0], self.hidden_dim), ms.float32)
        c = ops.Zeros()((1, ops.Shape()(x)[0], self.hidden_dim), ms.float32)
        x = self.dense1(x)
        for layer in self.layers:
            x, (h, c) = layer(x, h, c, g)
        x = self.dense2(x)
        return x


class GeniePathLazy(GNNCell):
    """GeniePath Lazy"""

    def __init__(self, input_dim, output_dim, hidden_dim=16, num_layers=2, num_attn_head=1, residual=False):
        self.residual = residual
        super(GeniePathLazy, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense1 = nn.Dense(input_dim, hidden_dim)
        self.dense2 = nn.Dense(hidden_dim, output_dim)
        self.breaths = nn.CellList()
        self.depths = nn.CellList()
        for _ in range(num_layers):
            self.breaths.append(GATConv(hidden_dim, hidden_dim, num_attn_head=num_attn_head))
            self.depths.append(nn.LSTM(hidden_dim * 2, hidden_dim))
        self.tanh = nn.Tanh()
        self.expand = ops.ExpandDims()
        self.cat = ops.Concat(-1)

    def construct(self, x, g: Graph):
        """construct function"""
        h = ops.Zeros()((1, ops.Shape()(x)[0], self.hidden_dim), ms.float32)
        c = ops.Zeros()((1, ops.Shape()(x)[0], self.hidden_dim), ms.float32)
        x = self.dense1(x)
        h_tmps = []
        for breath in self.breaths:
            h_tmp = breath(x, g)
            h_tmp = self.tanh(h_tmp)
            h_tmp = self.expand(h_tmp, 0)
            h_tmps.append(h_tmp)
        x = self.expand(x, 0)
        for h_tmp, depth in zip(h_tmps, self.depths):
            in_cat = self.cat((h_tmp, x))
            x, (h, c) = depth(in_cat, (h, c))
        x = x[0]
        x = self.dense2(x)
        return x
