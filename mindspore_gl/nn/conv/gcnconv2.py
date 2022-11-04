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
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class GCNConv2(GNNCell):
    r"""
    Graph Convolution Network Layer.
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
        - **g** (Graph) - The input graph.

    Outputs:
        Tensor, output node features with shape of :math:`(N, D_{out})`, where :math:`(D_{out})` should be the same as
        `out_size` in `Args`.

    Raises:
        TypeError: If `in_feat_size` or `out_size` is not an int.

    Supported Platforms:
         ``GPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import GCNConv2
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> gcnconv2 = GCNConv2(in_feat_size=4, out_size=2)
        >>> res = gcnconv2(feat, *graph_field.get_graph())
        >>> print(res.shape)
        (4, 2)
    """

    def __init__(self,
                 in_feat_size: int,
                 out_size: int) -> None:
        super().__init__()
        self.in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        self.out_size = Validator.check_positive_int(out_size, "out_size", self.cls_name)
        self.fc1 = ms.nn.Dense(in_feat_size, out_size, weight_init=XavierUniform(), has_bias=False)
        self.bias = ms.Parameter(initializer('zero', (out_size), ms.float32), name="bias")
        self.fc2 = ms.nn.Dense(in_feat_size, out_size, weight_init=XavierUniform(), has_bias=False)

    # pylint: disable=arguments-differ
    def construct(self, x, g: Graph):
        """
        Construct function for GCNConv.
        """
        x = ms.ops.Squeeze()(x)
        x_r = x
        x = self.fc1(x)
        g.set_vertex_attr({"x": x})
        for v in g.dst_vertex:
            v.x = g.sum([u.x for u in v.innbs])
            v.x += self.bias
        x = [v.x for v in g.dst_vertex]
        x = self.fc2(x_r) + x
        return x
