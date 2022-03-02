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
from mindspore_gl import Graph
from .. import GNNCell


class ShiftedSoftplus(ms.nn.Cell):
    """Shifted soft plus."""

    def __init__(self, shift=2.) -> None:
        super().__init__()
        self.shift = ms.Tensor([shift], ms.float32)
        self.softplus = ms.ops.Softplus()

    # pylint: disable=arguments-differ
    def construct(self, x):
        """
        Construct function for ShiftedSoftplus.

        Args:
            x (Tensor): The input node features.

        Returns:
            Tensor, output node features.
        """
        return self.softplus(x) - ms.ops.Log()(self.shift)


class CFConv(GNNCell):
    r"""
    CFConv in SchNet.
    From the paper `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions <https://arxiv.org/abs/1706.08566>`_.
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
    """

    def __init__(self,
                 node_feat_size: int,
                 edge_feat_size: int,
                 hidden_size: int,
                 out_size: int
                 ):
        super().__init__()
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

        Args:
            x (Tensor): The input node features.
            edge_feats (Tensor): The input edge features.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        g.set_vertex_attr({"hv": self.node_embedding_layer(x)})
        g.set_edge_attr({"he": self.edge_embedding_layer(edge_feats)})
        for v in g.dst_vertex:
            v.h = g.sum([s.hv * e.he for s, e in v.inedges])
        return self.out_embedding_layer([v.h for v in g.dst_vertex])
