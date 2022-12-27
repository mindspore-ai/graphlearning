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
"""Rgcn Net"""
import math
from typing import List
import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform

from mindspore_gl.nn import GNNCell
from mindspore_gl import Graph, HeterGraph


class HomoRGCNConv(GNNCell):
    """homo rgcn conv"""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 activation: callable = None):
        super().__init__()
        gain = math.sqrt(2)
        self.w = ms.Parameter(initializer(XavierUniform(gain), [in_size, out_size], ms.float32), name="w")
        self.act = activation

    def construct(self, x, in_deg, g: Graph):
        """homo rgcn conv forward"""
        g.set_vertex_attr(
            {"d": ms.ops.Reshape()(in_deg, ms.ops.Shape()(in_deg) + (1,)), "h": ms.ops.MatMul()(x, self.w)})
        for v in g.dst_vertex:
            v.h = g.sum([u.h for u in v.innbs]) / v.d
        ret = [v.h for v in g.dst_vertex]
        if self.act is not None:
            ret = self.act(ret)
        return ret


class RGCN(GNNCell):
    """Rgcn Net"""

    def __init__(self,
                 num_node_types: int,
                 cannonical_etypes: List[int],
                 input_size: int,
                 hidden_size: int,
                 output_size: int) -> None:
        super().__init__()
        self.can_etypes = cannonical_etypes
        self.n_types = num_node_types
        self.layer1 = HomoRGCNConv(input_size, hidden_size, ms.nn.LeakyReLU(0.01))
        self.layer2 = HomoRGCNConv(hidden_size, output_size, None)

    def construct(self, h, out_id, in_deg, hg: HeterGraph):
        """Rgcn Net forward"""
        new_h = []
        out = []
        for _ in range(self.n_types):
            new_h.append(ms.ops.Zeros()((1,), ms.float32))
            out.append(ms.ops.Zeros()((1,), ms.float32))
        for src_type, edge_type, dst_type in self.can_etypes:
            new_h[dst_type] += self.layer1(h[src_type], in_deg[edge_type], *hg.get_homo_graph(edge_type))
        for src_type, edge_type, dst_type in self.can_etypes:
            out[dst_type] += self.layer2(new_h[src_type], in_deg[edge_type], *hg.get_homo_graph(edge_type))
        return out[out_id]
