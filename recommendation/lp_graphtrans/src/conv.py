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
"""GCNConv Layer, GINConv Layer"""
import mindspore as ms
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
from mindspore.nn.cell import Cell
from mindspore_gl.nn import GNNCell
from mindspore_gl import BatchedGraph


class GCNConv(GNNCell):
    r"""
    Graph Convolution Network Layer.

    Args:
        GNNCell (Cell): cell layer
    """

    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 activation=None,
                 dropout=0.5) -> None:
        super().__init__()
        assert isinstance(in_feat_size, int) and in_feat_size > 0, "in_feat_size must be a positive int"
        assert isinstance(out_size, int) and out_size > 0, "out_size must be a positive int"
        assert isinstance(dropout, float), "dropout must be float"
        self.in_feat_size = in_feat_size
        self.out_size = out_size
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"For '{self.cls_name}', the 'dropout prob' should be a number in range [0.0, 1.0), "
                             f"but got {dropout}.")
        if activation is not None and not isinstance(activation, Cell):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must a Cell, but got "
                            f"{type(activation).__name__}.")
        self.fc = ms.nn.Dense(in_feat_size, out_size, weight_init=XavierUniform(), has_bias=False)
        self.bias = ms.Parameter(initializer('zero', (out_size), ms.float32), name="bias")
        self.activation = activation
        self.min_clip = Tensor(1, ms.int32)
        self.max_clip = Tensor(100000000, ms.int32)
        self.drop_out = ms.nn.Dropout(p=dropout)

    # pylint: disable=arguments-differ
    def construct(self, x, in_deg, out_deg, g: BatchedGraph):
        """
        Construct function for GCNConv.
        """
        out_deg = ms.ops.clip_by_value(out_deg, self.min_clip, self.max_clip)
        out_deg = ms.ops.Reshape()(ms.ops.Pow()(out_deg, -0.5), ms.ops.Shape()(out_deg) + (1,))
        x = self.drop_out(x)
        x = ms.ops.Squeeze()(x)
        x = x * out_deg
        x = self.fc(x)
        g.set_vertex_attr({"x": x})
        for v in g.dst_vertex:
            v.x = g.sum([u.x for u in v.innbs])
        in_deg = ms.ops.clip_by_value(in_deg, self.min_clip, self.max_clip)
        in_deg = ms.ops.Reshape()(ms.ops.Pow()(in_deg, -0.5), ms.ops.Shape()(in_deg) + (1,))
        x = [v.x for v in g.dst_vertex] * in_deg
        x = x + self.bias
        return x


class GINConv(GNNCell):
    r"""
    Graph isomorphic network layer.

    Args:
        GNNCell (Cell): cell layer
    """

    def __init__(self,
                 activation: ms.nn.Cell,
                 init_eps=0.,
                 learn_eps=False,
                 aggregation_type="sum"):
        super().__init__()
        self.agg_type = aggregation_type
        if aggregation_type not in {"sum", "max", "avg"}:
            raise SyntaxError("Aggregation type must be one of sum, max or avg")
        if learn_eps:
            self.eps = ms.Parameter(ms.Tensor(init_eps, ms.float32))
        else:
            self.eps = ms.Tensor(init_eps, ms.float32)
        self.act = activation

    def construct(self, x, edge_weight, g: BatchedGraph):
        """
        Construct function for GINConv.

        Args:
            x (Tensor): The input node features.
            edge_weight (Tensor): Edge weights.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        g.set_vertex_attr({"h": x})
        g.set_edge_attr({"w": edge_weight})
        for v in g.dst_vertex:
            if self.agg_type == 'sum':
                ret = g.sum([s.h * e.w for s, e in v.inedges])
            elif self.agg_type == 'max':
                ret = g.max([s.h * e.w for s, e in v.inedges])
            else:
                ret = g.avg([s.h * e.w for s, e in v.inedges])
            v.h = (1 + self.eps) * v.h + ret
            if self.act is not None:
                v.h = self.act(v.h)
        return [v.h for v in g.dst_vertex]
