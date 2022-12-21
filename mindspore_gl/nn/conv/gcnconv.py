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
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
from mindspore.nn.cell import Cell
from mindspore._checkparam import Validator
from mindspore_gl import Graph
from .. import GNNCell


class GCNConv(GNNCell):
    r"""
    Graph Convolution Network Layer.
    from the paper `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`_ .

    .. math::
        h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    :math:`\mathcal{N}(i)` represents the neighbour node of :math:`i`.
    :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`.

    .. math::
        h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{e_{ji}}{c_{ji}}h_j^{(l)}W^{(l)})

    Args:
        in_feat_size (int): Input node feature size.
        out_size (int): Output node feature size.
        activation (Cell): Activation function. Default: None.
        dropout (float): The keep rate, greater than 0 and less equal than 1. E.g. dropout=0.9,
            dropping out 10% of input units. Default: 0.5.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N, D_{in})`
          where :math:`N` is the number of nodes,
          and :math:`D_{in}` should be equal to `in_feat_size` in `Args`.
        - **in_deg** (Tensor) - In degree for nodes. The shape is :math:`(N, )` where :math:`N` is the number of nodes.
        - **out_deg** (Tensor) - Out degree for nodes. The shape is :math:`(N, )`
          where :math:`N` is the number of nodes.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, output node features with shape of :math:`(N, D_{out})`, where :math:`(D_{out})` should be the same as
          `out_size` in `Args`.

    Raises:
        TypeError: If `in_feat_size` or `out_size` is not an int.
        TypeError: If `dropout` is not a float.
        TypeError: If `activation` is not a mindspore.nn.Cell.
        ValueError: If `dropout` is not in range (0.0, 1.0]

    Supported Platforms:
        ``GPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import GCNConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> in_degree = ms.Tensor([3, 2, 1, 1], ms.int32)
        >>> out_degree = ms.Tensor([1, 2, 1, 2], ms.int32)
        >>> gcnconv = GCNConv(in_feat_size=4, out_size=2, activation=None, dropout=1.0)
        >>> res = gcnconv(feat, in_degree, out_degree, *graph_field.get_graph())
        >>> print(res.shape)
        (4, 2)
    """
    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 activation=None,
                 dropout=0.5) -> None:
        super().__init__()
        self.in_feat_size = Validator.check_positive_int(in_feat_size, "in_feat_size", self.cls_name)
        self.out_size = Validator.check_positive_int(out_size, "out_size", self.cls_name)
        dropout = Validator.check_is_float(dropout, "dropout", self.cls_name)
        if dropout <= 0.0 or dropout > 1.0:
            raise ValueError(f"For '{self.cls_name}', the 'keep_prob' should be a number in range (0.0, 1.0], "
                             f"but got {dropout}.")
        if activation is not None and not isinstance(activation, Cell):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must a mindspore.nn.Cell, but got "
                            f"{type(activation).__name__}.")
        self.fc = ms.nn.Dense(in_feat_size, out_size, weight_init=XavierUniform(), has_bias=False)
        self.bias = ms.Parameter(initializer('zero', (out_size), ms.float32), name="bias")
        self.activation = activation
        self.min_clip = Tensor(1, ms.int32)
        self.max_clip = Tensor(100000000, ms.int32)
        self.drop_out = ms.nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def construct(self, x, in_deg, out_deg, g: Graph):
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
