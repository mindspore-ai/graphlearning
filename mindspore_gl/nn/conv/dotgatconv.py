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
"""DOTGATConv Layer"""
import mindspore as ms
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class DOTGATConv(GNNCell):
    r"""
    Applying a dot product version of self-attention in GAT.
    From the paper `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`_ .

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i, j} h_j^{(l)}

    :math:`\alpha_{i, j}` represents the attention score between node :math:`i` and node :math:`j`.

    .. math::
        \alpha_{i, j} = \mathrm{softmax_i}(e_{ij}^{l}) \\
        e_{ij}^{l} = ({W_i^{(l)} h_i^{(l)}})^T \cdot {W_j^{(l)} h_j^{(l)}}

    Args:
        in_feat_size (int): Input node feature size.
        out_feat_size (int): Output node feature size.
        num_heads (int): Number of attention head used in GAT.
        bias (bool): Whether use bias. Default: False.

    Inputs:
        - **x** (Tensor): The input node features. The shape is :math:`(N,*)` where :math:`N` is the number of nodes,
          and :math:`*` could be of any shape.
        - **g** (Graph): The input graph.

    Outputs:
        - Tensor, output node features. The shape is :math:`(N, num_heads, out_feat_size)`.

    Raises:
        TypeError: If 'in_feat_size' is not a positive int.
        TypeError: If 'out_feat_size' is not a positive int.
        TypeError: If 'num_heads' is not a positive int.
        TypeError: If 'bias' is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import DOTGATConv
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
        >>> conv = DOTGATConv(feat_size, out_size, num_heads=2, bias=True)
        >>> ret = conv(nodes_feat, *graph_field.get_graph())
        >>> print(ret.shape)
        (4, 2, 4)
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 num_heads: int,
                 bias=False):
        super().__init__()
        in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        out_feat_size = Validator.check_positive_int(out_feat_size, "out_feat_size", self.cls_name)
        num_heads = Validator.check_positive_int(num_heads, "num_heads", self.cls_name)
        bias = Validator.check_bool(bias, 'bias', self.cls_name)

        self.dense = ms.nn.Dense(in_feat_size, out_feat_size * num_heads, has_bias=bias)
        self.num_heads = num_heads
        self.out_feat_size = out_feat_size

    # pylint: disable=arguments-differ
    def construct(self, x, g: Graph):
        """
        Construct function for DOTGATConv.
        """
        feat_src = feat_dst = ms.ops.Reshape()(self.dense(x), (-1, self.num_heads, self.out_feat_size))
        g.set_vertex_attr({"hsrc": feat_src, "hdst": feat_dst})
        for v in g.dst_vertex:
            dotted = [g.dot(u.hsrc, v.hdst) for u in v.innbs]
            a = dotted / g.sum(dotted)
            v.h = g.sum(a * [u.hsrc for u in v.innbs])
        return [v.h for v in g.dst_vertex]
