# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""SAGEConv Layer."""
import math
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import XavierUniform
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell

class SAGEConv(GNNCell):
    r"""
    GraphSAGE Layer. From the paper `Inductive Representation Learning on Large Graphs
    <https://arxiv.org/pdf/1706.02216.pdf>`_ .

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right) \\

        h_{i}^{(l+1)} = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)\\

        h_{i}^{(l+1)} = \mathrm{norm}(h_{i}^{l})

    If weights are provided on each edge, the weighted graph convolution is defined as:

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{e_{ji} h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        aggregator_type (str): Type of aggregator, should in 'pool', 'lstm' and 'mean'. Default: 'pool'.
        bias (bool): Whether use bias. Default: True.
        norm (Cell): Normalization function Cell. Default: None.
        activation (Cell): Activation function Cell. Default: None.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N,D\_in)`
          where :math:`N` is the number of nodes and :math:`D\_in` could be of any shape.
        - **edge_weight** (Tensor) - Edge weights. The shape is :math:`(N\_e,)`
          where :math:`N\_e` is the number of edges.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, the output feature of shape :math:`(N,D\_out)`.
          where :math:`N` is the number of nodes and :math:`D\_out` could be of any shape.

    Raises:
        TypeError: If `in_feat_size` or `out_feat_size` is not an int.
        TypeError: If `bias` is not a bool.
        KeyError: if `aggregator` type is not 'pool', 'lstm' or 'mean'.
        TypeError: if `activation` type is not mindspore.nn.Cell
        TypeError: if `norm` type is not mindspore.nn.Cell

    Supported Platforms:
        ``GPU`` ``Ascend``

    Examples:
       >>> import mindspore as ms
       >>> from mindspore import nn
       >>> from mindspore.numpy import ones
       >>> from mindspore_gl.nn import SAGEConv
       >>> from mindspore_gl import GraphField
       >>> n_nodes = 4
       >>> n_edges = 7
       >>> feat_size = 4
       >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
       >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
       >>> ones = ms.ops.Ones()
       >>> feat = ones((n_nodes, feat_size), ms.float32)
       >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
       >>> sageconv = SAGEConv(in_feat_size=4, out_feat_size=2, activation=nn.ReLU())
       >>> edge_weight = ones((n_edges, 1), ms.float32)
       >>> res = sageconv(feat, edge_weight, *graph_field.get_graph())
       >>> print(res.shape)
        (4,2)
    """
    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 aggregator_type: str = "pool",
                 bias=True,
                 norm=None,
                 activation: ms.nn.Cell = None):
        super().__init__()
        self.in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        self.out_feat_size = Validator.check_positive_int(out_feat_size, "out_feat_size", self.cls_name)
        self.agg_type = Validator.check_string(aggregator_type, ["mean", "pool", "lstm"], self.cls_name)
        bias = Validator.check_bool(bias, "bias", self.cls_name)

        if activation:
            if not isinstance(activation, nn.Cell):
                raise TypeError("activation type should be ms.nn.Cell")

        if norm:
            if not isinstance(norm, nn.Cell):
                raise TypeError("norm type should be ms.nn.Cell")

        self.in_feat_size = in_feat_size
        self.out_feat_size = out_feat_size
        self.agg_type = aggregator_type
        self.norm = norm
        self.activation = activation
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
        elif self.agg_type != 'mean':
            raise KeyError("Unknown aggregator type {}".format_map(self.agg_type))
        self.dense_self = ms.nn.Dense(self.in_feat_size, self.out_feat_size, has_bias=False,
                                      weight_init=XavierUniform(math.sqrt(2)))

    # pylint: disable=arguments-differ
    def construct(self, node_feat, edge_weight, g: Graph):
        """
        Construct function for SAGEConv.
        """
        g.set_vertex_attr({"h": node_feat})
        if self.agg_type == "mean" and self.in_feat_size > self.out_feat_size:
            g.set_vertex_attr({"h": self.dense_neigh(node_feat)})
        if self.agg_type == "pool":
            g.set_vertex_attr({"h": ms.ops.ReLU()(self.fc_pool(node_feat))})

        for v in g.dst_vertex:
            if edge_weight is not None:
                g.set_edge_attr({"w": edge_weight})
                neigh_feat = [s.h * e.w for s, e in v.inedges]
            else:
                neigh_feat = [u.h for u in v.innbs]
            if self.agg_type == "mean":
                v.h = g.avg(neigh_feat)
                if self.in_feat_size <= self.out_feat_size:
                    v.h = self.dense_neigh(v.h)
            if self.agg_type == "pool":
                v.h = g.max(neigh_feat)
                v.h = self.dense_neigh(v.h)
            if self.agg_type == "lstm":
                init_h = (ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),
                          ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32))
                _, (v.h, _) = self.lstm(neigh_feat, init_h)
                v.h = self.dense_neigh(ms.ops.Squeeze()(v.h, 0))
        out_feat = [v.h for v in g.dst_vertex]
        ret = self.dense_self(node_feat) + out_feat

        if self.bias is not None:
            ret = ret + self.bias

        if self.activation is not None:
            ret = self.activation(ret)

        if self.norm is not None:
            ret = self.norm(self.ret)

        return ret
        