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
"""CFConv Layer"""
import mindspore as ms
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class ShiftedSoftplus(ms.nn.Cell):
    """Shifted soft plus."""

    def __init__(self, shift=2.):
        super().__init__()
        self.shift = ms.Tensor([shift], ms.float32)
        self.softplus = ms.ops.Softplus()

    # pylint: disable=arguments-differ
    def construct(self, x):
        """
        Construct function for ShiftedSoftplus.
        """
        return self.softplus(x) - ms.ops.Log()(self.shift)


class CFConv(GNNCell):
    r"""
    CFConv in SchNet.
    From the paper `SchNet: A continuous-filter convolutional neural network for modeling quantum
    interactions <https://arxiv.org/abs/1706.08566>`_ .

    It combines node and edge features in messaging and updates node representations.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} h_j^{l} \circ W^{(l)}e_ij

    Where :math:`SPP` represents:

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    Args:
        node_feat_size (int): Node feature size.
        edge_feat_size (int): Edge feature size.
        hidden_size (int): Hidden layer size.
        out_size (int): Output classes size.

    Inputs:
        - **x** (Tensor): The input node features. The shape is :math:`(N,*)` where :math:`N` is the number of nodes,
          and :math:`*` could be of any shape.
        - **edge_feats** (Tensor): The input edge features. The shape is :math:`(M,*)` where :math:`M` is the number of
          edges, and :math:`*` could be of any shape.
        - **g** (Graph): The input graph.

    Outputs:
        - Tensor, output node features. The shape is :math:`(N, out\_size)`.

    Raises:
        TypeError: If 'node_feat_size' is not a positive int.
        TypeError: If 'edge_feat_size' is not a positive int.
        TypeError: If 'hidden_size' is not a positive int.
        TypeError: If 'out_size' is not a positive int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import CFConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 8
        >>> feat_size = 16
        >>> src_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 2, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 1, 3, 1, 2, 3, 3, 2], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> nodes_feat = ones((n_nodes, feat_size), ms.float32)
        >>> edges_feat = ones((n_edges, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> hidden_size = 8
        >>> out_size = 4
        >>> conv = CFConv(feat_size, feat_size, hidden_size, out_size)
        >>> ret = conv(nodes_feat, edges_feat, *graph_field.get_graph())
        >>> print(ret.shape)
        (4, 4)
    """

    def __init__(self,
                 node_feat_size: int,
                 edge_feat_size: int,
                 hidden_size: int,
                 out_size: int
                 ):
        super().__init__()
        node_feat_size = Validator.check_positive_int(node_feat_size, "node_feat_size", self.cls_name)
        edge_feat_size = Validator.check_positive_int(edge_feat_size, "edge_feat_size", self.cls_name)
        hidden_size = Validator.check_positive_int(hidden_size, "hidden_size", self.cls_name)
        out_size = Validator.check_positive_int(out_size, "out_size", self.cls_name)

        self.edge_embedding_layer = ms.nn.SequentialCell(
            ms.nn.Dense(edge_feat_size, hidden_size),
            ShiftedSoftplus(),
            ms.nn.Dense(hidden_size, hidden_size),
            ShiftedSoftplus()
        )

        self.node_embedding_layer = ms.nn.Dense(node_feat_size, hidden_size)

        self.out_embedding_layer = ms.nn.SequentialCell(
            ms.nn.Dense(hidden_size, out_size),
            ShiftedSoftplus()
        )

    # pylint: disable=arguments-differ
    def construct(self, x, edge_feats, g: Graph):
        """
        Construct function for CFConv.
        """
        g.set_vertex_attr({"hv": self.node_embedding_layer(x)})
        g.set_edge_attr({"he": self.edge_embedding_layer(edge_feats)})
        for v in g.dst_vertex:
            v.h = g.sum([s.hv * e.he for s, e in v.inedges])
        return self.out_embedding_layer([v.h for v in g.dst_vertex])
