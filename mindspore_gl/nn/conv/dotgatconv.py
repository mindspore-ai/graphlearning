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
from mindspore_gl import Graph
from .. import GNNCell


class DOTGATConv(GNNCell):
    r"""
    Applying a dot product version of self-attention in GCN.

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
        bias (bool): Whether use bias.
    """

    def __init__(self,
                 in_feat_size: int,
                 out_feat_size: int,
                 num_heads: int,
                 bias=False):
        super().__init__()
        self.dense = ms.nn.Dense(in_feat_size, out_feat_size * num_heads, has_bias=bias)
        self.num_heads = num_heads
        self.out_feat_size = out_feat_size

    # pylint: disable=arguments-differ
    def construct(self, x, g: Graph):
        """
        Construct function for DOTGATConv.

        Args:
            x (Tensor): The input node features.
            g (Graph): The input graph.

        Returns:
            Tensor, output node features.
        """
        feat_src = feat_dst = ms.ops.Reshape()(self.dense(x), (-1, self.num_heads, self.out_feat_size))
        g.set_vertex_attr({"hsrc": feat_src, "hdst": feat_dst})
        for v in g.dst_vertex:
            dotted = [g.dot(u.hsrc, v.hdst) for u in v.innbs]
            a = dotted / g.sum(dotted)
            v.h = g.sum(a * [u.hsrc for u in v.innbs])
        return [v.h for v in g.dst_vertex]
