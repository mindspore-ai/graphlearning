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
"""SAGEConv Layer."""
import math
import mindspore as ms
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from .. import GNNCell


class SAGEConv(GNNCell):
    r"""
    GraphSAGE Layer, from the paper `Inductive Representation Learning on Large Graphs
    <https://arxiv.org/pdf/1706.02216.pdf>`_.

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
        aggregator_type (str): Type of aggregator, should in 'pool', 'lstm' and 'mean'.
        feat_drop (float): Feature drop out rate.
        bias (bool): Whether use bias.
        norm (Cell): Normalization function Cell, default is None.
        activation (Cell): Activation function Cell, default is None.

    Raises:
        SyntaxError: when aggregator type not supported.
    """

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
        elif self.agg_type != 'mean':
            raise KeyError("Unknown aggregator type {}".format_map(self.agg_type))
        self.dense_self = ms.nn.Dense(self.in_feat_size, self.out_feat_size, has_bias=False,
                                      weight_init=XavierUniform(math.sqrt(2)))

    # pylint: disable=arguments-differ
    def construct(self, node_feat, edge_weight, g: Graph):
        """
        Construct function for SAGEConv.

        Args:
            node_feat (Tensor): The input node features.
            edge_weight (Tensor): Edge weights.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        node_feat = self.feat_drop(node_feat)

        if self.agg_type == "mean" and self.in_feat_size > self.out_feat_size:
            node_feat = self.dense_neigh(node_feat)
        if self.agg_type == "pool":
            node_feat = ms.ops.ReLU()(self.fc_pool(node_feat))
        g.set_vertex_attr({"h": node_feat})

        for v in g.dst_vertex:
            if edge_weight is not None:
                g.set_edge_attr({"w": edge_weight})
                v.rst = [s.h * e.w for s, e in v.inedges]
            else:
                v.rst = [u.h for u in v.innbs]
            if self.agg_type == "mean":
                v.rst = g.avg(v.rst)
                if self.in_feat_size <= self.out_feat_size:
                    v.rst = self.dense_neigh(v.rst)
            if self.agg_type == "pool":
                v.rst = g.max(v.rst)
                v.rst = self.dense_neigh(v.rst)
            if self.agg_type == "lstm":
                init_h = (ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32),
                          ms.ops.Zeros()((1, g.n_edges, self.in_feat_size), ms.float32))
                _, (v.rst, _) = self.lstm(v.rst, init_h)
                v.rst = self.dense_neigh(ms.ops.Squeeze()(v.rst, 0))

        ret = self.dense_self(node_feat) + [v.rst for v in g.dst_vertex]

        if self.bias is not None:
            ret = ret + self.bias

        if self.activation is not None:
            ret = self.activation(ret)

        if self.norm is not None:
            ret = self.norm(self.ret)

        return ret
