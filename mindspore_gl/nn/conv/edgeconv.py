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
"""EDGEConv Layer"""
import mindspore as ms
from mindspore_gl import Graph
from .. import GNNCell


class EDGEConv(GNNCell):
    r"""
    EdgeConv layer. From the paper `Dynamic Graph CNN for Learning on Point Clouds <https://arxiv.org/pdf/1801.07829>`_ .

    .. math::
        h_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} (
       \Theta \cdot (h_j^{(l)} - h_i^{(l)}) + \Phi \cdot h_i^{(l)})

    :math:`\mathcal{N}(i)` represents the neighbour node of :math:`i`.
    :math:`\Theta` and :math:`\Phi` represents linear layers.

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        batch_norm (bool): Whether use batch norm.
        bias (bool, optional): Whether use bias. Default: ``True``.

    Inputs:
        - **x** (Tensor): The input node features. The shape is :math:`(N,*)` where :math:`N` is the number of nodes,
          and :math:`*` could be of any shape.
        - **g** (Graph): The input graph.

    Outputs:
        - Tensor, output node features. The shape is :math:`(N, out\_feat\_size)`.

    Raises:
        TypeError: If `in_feat_size` is not a positive int.
        TypeError: If `out_feat_size` is not a positive int.
        TypeError: If `batch_norm` is not a bool.
        TypeError: If `bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import EDGEConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 8
        >>> feat_size = 16
        >>> src_idx = ms.Tensor([0, 0, 0, 1, 1, 1, 2, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 1, 3, 1, 2, 3, 3, 2], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> nodes_feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> out_size = 4
        >>> conv = EDGEConv(feat_size, out_size, batch_norm=True)
        >>> ret = conv(nodes_feat, *graph_field.get_graph())
        >>> print(ret.shape)
        (4, 4)
    """

    # pylint: disable=arguments-differ
    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 batch_norm: bool,
                 bias=True):
        super().__init__()
        assert isinstance(in_feat_size, int) and in_feat_size > 0, "in_feat_size must be positive int"
        assert isinstance(out_feat_size, int) and out_feat_size > 0, "out_feat_size must be positive int"
        assert isinstance(batch_norm, bool) and batch_norm > 0, "batch_norm must be bool"
        assert isinstance(bias, bool), "bias must be bool"

        self.batch_norm = batch_norm
        self.theta = ms.nn.Dense(in_feat_size, out_feat_size, has_bias=bias)
        self.phi = ms.nn.Dense(in_feat_size, out_feat_size, has_bias=bias)
        if batch_norm:
            self.bn = ms.nn.BatchNorm1d(out_feat_size)

    def construct(self, x, g: Graph):
        """
        Construct function for EDGEConv.
        """
        g.set_vertex_attr({"x": x, "phi": self.phi(x)})
        for v in g.dst_vertex:
            if not self.batch_norm:
                v.h = g.max([self.theta(u.x - v.x) + v.phi for u in v.innbs])
            else:
                v.h = g.max([self.bn(self.theta(u.x - v.x) + v.phi) for u in v.innbs])
        return [v.h for v in g.dst_vertex]
