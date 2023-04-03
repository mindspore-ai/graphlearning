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
"""GCNConv Layer"""
import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
from mindspore_gl import Graph
from .. import GNNCell


class GCNEConv(GNNCell):
    r"""
    Graph Convolution Network Layer with Edge Weight.
    from the paper `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`_.

    .. math::
        `h_i^{(l+1)} = (\sum_{j\in\mathcal{N}(i)}h_j^{(l)}W_1^{(l)}+b^{(l)} )+h_i^{(l)}W_2^{(l)}`

    :math:`\mathcal{N}(i)` represents the neighbour node of :math:`i`.
    :math:`W_1` and `W_2` correspond to fc layers for neighbor nodes and root node.

    Args:
        in_feat_size (int): Input node feature size.
        out_size (int): Output node feature size.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N, D_{in})`
          where :math:`N` is the number of nodes,
          and :math:`D_{in}` should be equal to `in_feat_size` in `Args`.
        - **edge_weight** (Tensor) - The weight of edges . The shape is :math:`(N, 1)`
          where :math:`N` is the number of nodes.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, output node features with shape of :math:`(N, D_{out})`, where :math:`(D_{out})` should be the same as
          `out_size` in `Args`.

    Raises:
        TypeError: If `in_feat_size` or `out_size` is not an int.

    Supported Platforms:
        ``GPU`` ``Ascend``

    Examples:
        >>> node_feat = ms.Tensor([[1, 2, 3, 4], [2, 4, 1, 3], [1, 3, 2, 4], [9, 7, 5, 8],
        ...                        [8, 7, 6, 5], [8, 6, 4, 6], [1, 2, 1, 1]], ms.float32)
        >>> n_nodes = 7
        >>> n_edges = 8
        >>> src_idx = ms.Tensor([0, 2, 2, 3, 4, 5, 5, 6], ms.int32)
        >>> dst_idx = ms.Tensor([1, 0, 1, 5, 3, 4, 6, 4], ms.int32)
        >>> edge_weight = ms.Tensor([[1], [1], [1], [1], [1], [1], [1], [1]], ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> net = GCNEConv(4, 1, True)
        >>> output = net(node_feat, edge_weight, *graph_field.get_graph())
        >>> print(output.shape)
        (7, 1)
    """
    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 bias: bool = False):
        super().__init__()
        assert isinstance(in_feat_size, int) and in_feat_size > 0, "in_feat_size must be positive int"
        assert isinstance(out_size, int) and out_size > 0, "out_size must be positive int"
        self.in_feat_size = in_feat_size
        self.out_size = out_size
        self.fc1 = ms.nn.Dense(in_feat_size, out_size, weight_init=XavierUniform(), has_bias=False)
        if bias:
            self.bias = ms.Parameter(initializer('zero', (out_size), ms.float32), name="bias")
        else:
            self.bias = None

    # pylint: disable=arguments-differ
    def construct(self, x, edge_weight, g: Graph):
        """
        Construct function for GCNConv.
        """
        x = ms.ops.Squeeze()(x)
        x = self.fc1(x)
        g.set_vertex_attr({"x": x})
        for v in g.dst_vertex:
            g.set_edge_attr({"w": edge_weight})
            v.x = g.sum([u.x * e.w for u, e in v.inedges])
        x = [v.x for v in g.dst_vertex]
        if self.bias is not None:
            x += self.bias
        return x
